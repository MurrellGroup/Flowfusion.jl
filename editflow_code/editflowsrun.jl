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
        # x0: uniform tokens with random length in 10:30 (no BOS in x0)
        L0 = rand(rng, 1:10)
        seq0 = rand(rng, 1:K, L0)
        x0s[b] = FF.DiscreteState(K, seq0)
    end
    ts = rand(rng, Float32, B)
    return x0s, x1s, ts
end

struct EditFlowModel{L}
    layers::L
end
Flux.@layer EditFlowModel

function EditFlowModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, cond_dim=128, K::Int)
    embedding   = Flux.Embedding(K + 2 => d)
    time_embed  = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim))
    blocks      = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    head_combined = Dense(d => 2K + 1, bias=false)
    rope        = RoPE(d ÷ num_heads, 4096)
    return EditFlowModel((; embedding, time_embed, blocks, head_combined, rope, K))
end

function (model::EditFlowModel)(t, Xt_ms)
    m = model.layers
    X = FF.tensor(Xt_ms)
    X = ndims(X) == 1 ? reshape(X, :, 1) : X
    L, B = size(X)

    pmask = Zygote.@ignore FF.getlmask(Xt_ms)
    Xp = X .+ 1
    H = m.embedding(Xp)

    t = ndims(t) == 0 ? fill(Float32(t), B) : Float32.(t)
    cond = m.time_embed(reshape(t, 1, B))

    cond  = to_same_device(cond, H)
    pmask = Zygote.@ignore to_same_device(pmask, H)
    rope  = Zygote.@ignore to_same_device(m.rope[1:L], H)

    for blk in m.blocks
        H = blk(H; cond, rope, kpad_mask=pmask)
    end
    return m.head_combined(H)
end

# Training (GPU if available)
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

    # Move model to device (GPU if available)
    model = Functors.fmap(to_dev, model)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    for epoch in 1:epochs
        for step in 1:steps_per_epoch

            # 1) Minibatch Sampling
            x0s, x1s, ts = make_minibatch(batch_size, P; rng=rng)

            # align_and_batch basically 
            Z0, Z1 = FF.align_and_batch(P, x0s, x1s)

            #
            Zt, Xt = FF.interpolate_Z_elementwise(P, Z0, Z1, ts)

            #Append BOS token to the beginning of the batch
            bos = P.bos_token
            Zt = vcat(fill(bos, 1, batch_size), Zt)
            Xt = vcat(fill(bos, 1, batch_size), Xt)
            Z1 = vcat(fill(bos, 1, batch_size), Z1)
            
            transition_mask = FF.transition_mask_from_Xt(P, Xt)
            edit_multiplier = FF.remaining_edits(P, Zt, Z1)

            den = 1f0 .- P.κ.(ts)
            den = max.(den, 1f-6)
            scheduler_scaling = P.dκ.(ts) ./ den

            # 3) Masked state (CPU → device)
            lmask = Xt .!= P.padding_token
            cmask = trues(size(lmask))
            Xt_ms = FF.MaskedState(FF.DiscreteState(P.k, Xt), cmask, lmask)

            ts_d    = to_dev(ts)
            Xt_ms_d = to_dev(Xt_ms)
            Tmask_d = to_dev(transition_mask)
            Emult_d = to_dev(edit_multiplier)
            sched_d = to_dev(reshape(Float32.(scheduler_scaling), 1, 1, :))
            #print("RE???")

            # 4) Forward + loss + update (on device)
            loss, grad = Flux.withgradient(model) do m
                M = m(ts_d, Xt_ms_d)
                l = FF.edit_loss(P, M, Tmask_d, Emult_d, sched_d; eps=1f-8)
                # INSERT_YOUR_CODE
                if isnan(l)
                    println("Maximum element of M: ", maximum(M))
                end
                l
            end
            Flux.update!(opt_state, model, grad[1])

            if step % print_every == 0
                @info "train2" epoch step loss=Float32(loss)
            end
        end
    end

    # Move model back to CPU for sampling
    return Functors.fmap(to_cpu, model)
end

# PM is the target probability model and K is the alphabet size
K = PM.K

P = FF.EditFlow(K; bos_token=0)

model = EditFlowModel(; d=128, num_heads=8, nlayers=4, rff_dim=128, cond_dim=128, K=K)

# Train; returned model is on CPU
model = train_editflow!(P, model; epochs=2, steps_per_epoch=150, batch_size=256, lr=1f-3)

rng = Random.MersenneTwister(42)
println("\n=== True PM samples (20) ===")
for i in 1:20
    seq = sample(PM; rng=rng)                 # Vector{Int} in 1..K_AA
    aa_str = String(collect(AA20[seq]))
    println("[", i, "] ", aa_str)
end

samples = sample_gen_10_strings(P, model; ts=0f0:0.01f0:1f0)

println("\n=== Model samples (10) ===")
for (i, s) in enumerate(samples)
    if s isa AbstractString
        println("[", i, "] ", s)
    elseif s isa AbstractVector{<:AbstractString}
        last_str = isempty(s) ? "" : s[end]
        println("[", i, "] traj_last=", last_str, " (len=", length(s), ")")
    else
        println("[", i, "] (raw) ", s)
    end
end