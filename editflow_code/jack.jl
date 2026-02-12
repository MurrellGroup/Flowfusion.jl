using Pkg
Pkg.activate(".")
ENV["CUDA_VISIBLE_DEVICES"] = 1
using Revise, Random, Statistics, Adapt, Functors, Flux, Onion, RandomFeatureMaps, Zygote, CannotWaitForTheseOptimisers, CodecZlib, CSV, LearningSchedules, DataFrames, Distributions
using Flowfusion
const FF = Flowfusion
include("scripts/model_training/editflows/prob_model.jl")
include("scripts/model_training/editflows/helper_funcs.jl")
include("scripts/misc/data_loading.jl")
import CUDA

# Device helpers
const _gpu_enabled = try
    CUDA.has_cuda()
catch
    false
end

to_dev(x) = _gpu_enabled ? Adapt.adapt(CUDA.CuArray, x) : x
to_cpu(x) = _gpu_enabled ? Adapt.adapt(Array, x) : x
to_same_device(x, y) = (_gpu_enabled && (y isa CUDA.CuArray)) ? Adapt.adapt(CUDA.CuArray, x) : Adapt.adapt(Array, x)

# Load PM target (brings PM and AA20)

struct EditFlowModel{L}
    layers::L
end
Flux.@layer EditFlowModel

function EditFlowModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, cond_dim=128, K::Int)
    embedding   = Flux.Embedding(K + 2 => d)
    time_embed  = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim))
    blocks      = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    head_combined = Dense(d => 2K + 1, bias=false)
    rope        = RoPE(d ÷ num_heads, 512)
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

data = readlines("data/oas_heavy_8M.txt")

K = PM.K
P = FF.EditFlow(K; bos_token=0)
epochs = 1; batch_size = 64

model = EditFlowModel(; d=512, num_heads=8, nlayers=12, rff_dim=512, cond_dim=512, K=K)
const START_LR     = 1f-3
const MAX_LR       = 1f-3
const UP_GAMMA     = 1.00f0
const DOWN_GAMMA   = 0.99994398f0

sched = burnin_learning_schedule(START_LR, MAX_LR, UP_GAMMA, DOWN_GAMMA)
for layer in model.layers.blocks
    layer.attention_norm.scale.weight .= 0.0f0
    layer.attention_norm.shift.weight .= 0.0f0
    layer.ffn_norm.scale.weight .= 0.0f0
    layer.ffn_norm.shift.weight .= 0.0f0
end

rng = Random.MersenneTwister(seed)
Random.seed!(seed)
global_step = 1
model = Functors.fmap(to_dev, model)
sched = burnin_learning_schedule(START_LR, MAX_LR, UP_GAMMA, DOWN_GAMMA)

# do learning rate warmdown 
total_iters = div(512000, batch_size)
DECAY_STEPS = 8000
get_lr(i) = 1f-3 + (1f-7 - 1f-3) * (Float32(min(i, DECAY_STEPS)) / Float32(DECAY_STEPS))
opt_state = Flux.setup(Muon(eta=get_lr(1)), model)

for epoch in 1:epochs
    # for (step, seqs) in enumerate(stream_heavy_batches(path; L=140, batch=batch_size, total=10^6))
    for step in 1:total_iters
        seqs = [rand(data) for _ in 1:batch_size]
        xpairs = format_for_editflows(seqs)
        x0s = [p[1] for p in xpairs]
        x1s = [p[2] for p in xpairs]
        ts = rand(Float32, batch_size)
        # align_and_batch basically
        Z0, Z1 = try
            FF.align_and_batch(P, x0s, x1s)
        catch e
            if e isa LoadError || e isa ArgumentError
                @warn "skip batch due to Data loading error" err=e
                continue
            else
                rethrow(e)
            end
        end
        Zt, Xt = FF.interpolate_Z_elementwise(P, Z0, Z1, ts)
        #Append BOS token to the beginning of the batch
        bos = P.bos_token
        Zt = vcat(fill(bos, 1, batch_size), Zt)
        Xt = vcat(fill(bos, 1, batch_size), Xt)
        Z1 = vcat(fill(bos, 1, batch_size), Z1)

        transition_mask = FF.transition_mask_from_Xt(P, Xt)
        edit_multiplier = FF.remaining_edits(P, Zt, Z1, Xt)
        den = (1f0 .- P.κ.(ts)) .+ 0.2f0
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
        loss, grad = try
            Flux.withgradient(model) do m
                M = m(ts_d, Xt_ms_d)
                FF.edit_loss(P, M, Tmask_d, Emult_d, sched_d; eps=1f-8)
            end
        catch e
            if e isa DimensionMismatch
                @warn "skip batch due to DimensionMismatch" err=e
                continue
            else
                rethrow(e)
            end
        end
        if Float32(loss) < 0
            @warn "negative loss at step $step"
            continue
        end
        open("losses/EDITFLOWS.csv", "a") do io
            println(io, "$(global_step),$(epoch),$(step),$(Float32(loss)),$(get_lr(step)),$(time())")
        end
        Flux.adjust!(opt_state, get_lr(step))
        Flux.update!(opt_state, model, grad[1])
        if step % 10 == 0
            @info "EditFlows" epoch step loss=Float32(loss)
        end
        if step % 500 == 0
            samples = sample_gen_10_strings(P, Functors.fmap(to_cpu, model), n=5)
            CSV.write("gens/samples_step_$(step).csv",DataFrame(step = fill(step, length(samples)), sample = samples))
        end
        global_step += 1
    end
end

model_name = "EDITFLOWS"
using JLD2
model_state = Flux.state(cpu(model))
model_state_file = "models/model_state_"*model_name*".jld2"
JLD2.@save model_state_file model_state

open("losses/loss_$(model_name).txt", "w") do io
    println.(io, losses)
end

rng = Random.MersenneTwister(42)
println("\n=== True PM samples (20) ===")
for i in 1:20
    seq = sample(PM; rng=rng)                 # Vector{Int} in 1..K_AA
    aa_str = String(collect(AA20[seq]))
    println("[", i, "] ", aa_str)
end

gens = sample_gen_10_strings(P, Functors.fmap(to_cpu, model); ts=0f0:0.005f0:1f0, n=500)
gens = replace.(gens, ">"=>"")
open("gens/$(model_name)_gens.txt", "w") do io
    println.(io, gens)
end

println("\n=== Model samples (10) ===")
for (i, s) in enumerate(gens)
    if s isa AbstractString
        println("[", i, "] ", s)
    elseif s isa AbstractVector{<:AbstractString}
        last_str = isempty(s) ? "" : s[end]
        println("[", i, "] traj_last=", last_str, " (len=", length(s), ")")
    else
        println("[", i, "] (raw) ", s)
    end
end
