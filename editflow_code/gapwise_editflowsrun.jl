using Pkg
Pkg.activate(@__DIR__)
#Pkg.instantiate()
using Revise

using Random
using Statistics
using Adapt
using Functors
using Flux
using Onion
using RandomFeatureMaps
using Zygote
Pkg.develop(path=joinpath(@__DIR__, ".."))
using Flowfusion
const FF = Flowfusion
include(joinpath(@__DIR__, "prob_model.jl"))
include(joinpath(@__DIR__, "helper_funcs.jl"))

import CUDA

# Device helpers
const _gpu_enabled = try
    CUDA.has_cuda()
catch
    false
end

to_dev(x) = _gpu_enabled ? Adapt.adapt(CUDA.CuArray, x) : x
to_cpu(x) = _gpu_enabled ? Adapt.adapt(Array, x) : x
# move x to the same device type as y (Array or CuArray)
to_same_device(x, y) = (_gpu_enabled && (y isa CUDA.CuArray)) ? Adapt.adapt(CUDA.CuArray, x) : Adapt.adapt(Array, x)

# Load PM target (brings PM and AA20)

function make_minibatch(B::Int, P::FF.EditFlow; rng=Random.default_rng())
    K = P.k
    x0s = Vector{FF.DiscreteState}(undef, B)
    x1s = Vector{FF.DiscreteState}(undef, B)
    for b in 1:B
        # x1 from true PM
        seq1 = sample(PM; rng=rng)
        @assert all(1 .<= seq1 .<= K)
        x1s[b] = FF.DiscreteState(K, seq1)
        # x0: uniform tokens with random length in 1:10 (no BOS in x0)
        L0 = rand(rng, 1:10)
        seq0 = rand(rng, 1:K, L0)
        x0s[b] = FF.DiscreteState(K, seq0)
    end
    ts = rand(rng, Float32, B)
    return x0s, x1s, ts
end


############################
# Gap-wise model + training
############################

# --- helper: build gap embeddings (pick one) ---
gaps_from_sites_dup(H) = hcat(@view(H[:,1:1,:]), @view(H[:,2:end,:]), @view(H[:,end:end,:]))


function gaps_from_sites_avg(H)
    d, L, B = size(H)
    if L == 1
        middle = @view H[:, 1:0, :]
    else
        @views middle = 0.5f0 .* (H[:, 1:L-1, :] .+ H[:, 2:L, :])
    end
    # cat med dims=2 är säkrare på GPU än hcat i vissa kombinationer av views
    return cat(@view(H[:, 1:1, :]), middle, @view(H[:, L:L, :]); dims=2) # d×(L+1)×B
end

struct EditFlowModel{L}
    layers::L
end
Flux.@layer EditFlowModel

function EditFlowModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, cond_dim=128, K::Int)
    embedding   = Flux.Embedding(K + 2 => d)
    time_embed  = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim))
    blocks      = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    # --- split heads: ins on gaps, sub/del on sites ---
    head_ins = Dense(d => K, bias=false)   # applied on gaps (L+1)
    head_sub = Dense(d => K, bias=false)   # applied on sites (L)
    head_del = Dense(d => 1, bias=false)   # applied on sites (L)
    rope        = RoPE(d ÷ num_heads, 4096)
    return EditFlowModel((; embedding, time_embed, blocks, head_ins, head_sub, head_del, rope, K))
end

# Forward: returns M of shape (2K+1, L+1, B)
function (model::EditFlowModel)(t, Xt_ms)
    m = model.layers
    X = FF.tensor(Xt_ms)
    X = ndims(X) == 1 ? reshape(X, :, 1) : X
    L, B = size(X)

    pmask = Zygote.@ignore FF.getlmask(Xt_ms)
    Xp = X .+ 1                     # embedding is 1-indexed
    H = m.embedding(Xp)             # d×L×B

    t = ndims(t) == 0 ? fill(Float32(t), B) : Float32.(t)
    cond = m.time_embed(reshape(t, 1, B))

    cond  = to_same_device(cond, H)
    pmask = Zygote.@ignore to_same_device(pmask, H)
    rope  = Zygote.@ignore to_same_device(m.rope[1:L], H)

    for blk in m.blocks
        H = blk(H; cond, rope, kpad_mask=pmask)   # d×L×B
    end

    # --- gap embeddings & heads ---
    Hg  = gaps_from_sites_avg(H)                  # d×(L+1)×B  (or use gaps_from_sites_dup)
    ins = m.head_ins(Hg)                          # K×(L+1)×B
    sub = m.head_sub(H)                           # K×L×B
    del = m.head_del(H)                           # 1×L×B

    # pad sub/del with a zero last column to reach L+1
    @views sub_pad = cat(sub, sub[:, 1:1, :].*0; dims=2)
    @views del_pad = cat(del, del[:, 1:1, :].*0; dims=2)

    # final combined logits (untransformed): (2K+1)×(L+1)×B
    return vcat(ins, sub_pad, del_pad)
end

# --- training loop (gap-wise) ---
function train_editflow!(P::FF.EditFlow,
                         model;
                         epochs::Int=1,
                         steps_per_epoch::Int=100,
                         batch_size::Int=64,
                         lr::Float32=1f-2,
                         seed::Int=42,
                         print_every::Int=25)

    rng = Random.MersenneTwister(seed)
    Random.seed!(seed)

    # device
    model = Functors.fmap(to_dev, model)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    for epoch in 1:epochs
        for step in 1:steps_per_epoch
            # 1) minibatch
            x0s, x1s, ts = make_minibatch(batch_size, P; rng=rng)
            Z0, Z1 = FF.align_and_batch(P, x0s, x1s)
            Zt, Xt = FF.interpolate_Z_elementwise(P, Z0, Z1, ts)

            # prepend BOS to all streams
            bos = P.bos_token
            Zt = vcat(fill(bos, 1, batch_size), Zt)
            Xt = vcat(fill(bos, 1, batch_size), Xt)
            Z1 = vcat(fill(bos, 1, batch_size), Z1)

            # 2) gap-wise masks & multipliers
            transition_mask = FF.transition_mask_from_Xt_gapwise(P, Xt)   # (2K+1, L+1, B)
            edit_multiplier = FF.remaining_edits_gapwise(P, Zt, Z1, Xt)   # (2K+1, L+1, B)
            @assert size(transition_mask) == size(edit_multiplier)
            @assert size(transition_mask, 2) == size(Xt, 1) + 1

            # scheduler
            den = 1f0 .- P.κ.(ts); den = max.(den, 1f-2)
            scheduler_scaling = P.dκ.(ts) ./ den  # length B

            # 3) masked state → device
            lmask = Xt .!= P.padding_token
            cmask = trues(size(lmask))
            Xt_ms = FF.MaskedState(FF.DiscreteState(P.k, Xt), cmask, lmask)

            ts_d    = to_dev(ts)
            Xt_ms_d = to_dev(Xt_ms)
            Tmask_d = to_dev(transition_mask)
            Emult_d = to_dev(edit_multiplier)
            sched_d = to_dev(reshape(Float32.(scheduler_scaling), 1, 1, :))

            # 4) fwd + loss + update
            loss, grad = Flux.withgradient(model) do m
                M = m(ts_d, Xt_ms_d)   # (2K+1, L+1, B)
                # shape sanity
                @assert size(M) == size(Tmask_d) == size(Emult_d)
                FF.edit_loss_gapwise(P, M, Tmask_d, Emult_d, sched_d; eps=1f-8)
            end
            Flux.update!(opt_state, model, grad[1])

            if step % print_every == 0
                @info "train-gapwise" epoch step loss=Float32(loss)
            end
        end
    end

    return Functors.fmap(to_cpu, model)
end

# PM is the target probability model and K is the alphabet size
K = PM.K
println("K: ", K)

P = FF.EditFlow(K; bos_token=0, impl="positionwise_reparam")

model = EditFlowModel(; d=128, num_heads=8, nlayers=4, rff_dim=128, cond_dim=128, K=K)

# Train; returned model is on CPU
model = train_editflow!(P, model; epochs=10, steps_per_epoch=150, batch_size=256, lr=1f-3)




