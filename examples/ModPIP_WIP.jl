using Pkg
Pkg.activate(@__DIR__)

using Flowfusion
using ForwardBackward
using Flux
using Optimisers
using NNlib
using Random
using Statistics

const AAS = collect("ACDEFGHIKLMNPQRSTVWY#-*")
const TOK2ID = Dict(c => i for (i, c) in enumerate(AAS))
const ID2TOK = Dict(v => k for (k, v) in TOK2ID)
const MASK_ID = TOK2ID['#']
const LEFT_IMMORTAL_ID = TOK2ID['-']
const RIGHT_IMMORTAL_ID = TOK2ID['*']
const NAA = 20
const KPROC = NAA + 1
const KMODEL = length(AAS)

encode(v::Vector{Int}) = join(ID2TOK[i] for i in v)

const USE_CUDA = let want = lowercase(get(ENV, "MODPIP_DEVICE", "auto")) != "cpu"
    if !want
        false
    else
        try
            @eval using CUDA
            CUDA.functional()
        catch
            false
        end
    end
end

if USE_CUDA
    to_device(x) = CUDA.cu(x)
    to_host(x) = CUDA.cpu(x)
    device_name() = CUDA.name(CUDA.device())
else
    to_device(x) = x
    to_host(x) = x
    device_name() = "cpu"
end

P = ModPIPProcess(KPROC; lambda = 0.35f0, mu = 0.35f0)

const FAMILY_MOTIFS = [
    [[2, 1, 2], [3, 4, 5, 6], [7, 2, 7]],
    [[8, 8, 9], [10, 11, 12], [13, 8, 13]],
    [[14, 15, 14], [16, 17, 18], [15, 14, 19]],
    [[20, 18, 20], [17, 16, 15], [20, 6, 20]],
]

const FAMILY_BG = [
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [5, 6, 15, 17, 18, 19, 20],
]

function sample_target(rng::AbstractRNG)
    family = rand(rng, 1:length(FAMILY_MOTIFS))
    motifs = FAMILY_MOTIFS[family]
    bg = FAMILY_BG[family]
    tokens = Int[]
    blocks = rand(rng, 2:5)
    for _ in 1:blocks
        append!(tokens, rand(rng, bg, rand(rng, 3:7)))
        motif = motifs[rand(rng, 1:length(motifs))]
        for tok in motif
            push!(tokens, rand(rng) < 0.12 ? rand(rng, bg) : tok)
        end
    end
    append!(tokens, rand(rng, bg, rand(rng, 2:6)))
    return tokens
end

const X0_PRIOR = lowercase(get(ENV, "MODPIP_X0_PRIOR", "random"))

function sample_x0(rng::AbstractRNG)
    len = rand(rng, 12:56)
    if X0_PRIOR == "mask"
        return fill(MASK_ID, len)
    elseif X0_PRIOR == "random"
        return rand(rng, 1:NAA, len)
    else
        error("Unsupported MODPIP_X0_PRIOR=$(X0_PRIOR). Use `random` or `mask`.")
    end
end

function sample_pair(rng::AbstractRNG)
    x0_tokens = sample_x0(rng)
    x1_tokens = sample_target(rng)
    return DiscreteState(KPROC, x0_tokens), DiscreteState(KPROC, x1_tokens)
end

struct ModPIPModel{L}
    layers::L
end

Flux.@layer ModPIPModel

function ModPIPModel(; d = 128, depth = 4, max_len = 192, K::Int)
    embedding = Flux.Embedding(K => d)
    pos_embedding = Flux.Embedding(max_len => d)
    time_embed = Flux.Chain(Dense(1 => d, Flux.gelu), Dense(d => d, Flux.gelu))
    trunk = Flux.Chain([Dense((i == 1 ? 3d : d) => d, Flux.gelu) for i in 1:depth]...)
    head_sub = Dense(d => K - 2, bias = false)
    head_del = Dense(d => 1, bias = false)
    head_ins_tok = Dense(d => K - 2, bias = false)
    head_ins_count = Dense(d => 1, bias = false)
    return ModPIPModel((; embedding, pos_embedding, time_embed, trunk,
                        head_sub, head_del, head_ins_tok, head_ins_count, max_len))
end

function (model::ModPIPModel)(t, Xt)
    m = model.layers
    toks = tensor(Xt)
    L, B = size(toks)
    @assert L <= m.max_len "Increase `max_len` in ModPIPModel."

    pos = Flux.Zygote.@ignore to_device(repeat(reshape(collect(1:L), L, 1), 1, B))
    Htok = m.embedding(toks) .+ m.pos_embedding(pos)
    Htime = repeat(reshape(m.time_embed(reshape(t, 1, B)), :, 1, B), 1, L, 1)
    Hglob = repeat(mean(Htok; dims = 2), 1, L, 1)
    H = m.trunk(cat(Htok, Htime, Hglob; dims = 1))

    Hgap = H[:, 1:end-1, :]
    Hreal = H[:, 2:end-1, :]

    return (
        sub = m.head_sub(Hreal),
        del = m.head_del(Hreal),
        ins_tok = m.head_ins_tok(Hgap),
        ins_count = m.head_ins_count(Hgap),
    )
end

function make_model_input(Xts)
    Flowfusion.batch(Flowfusion.prefix.(Xts, LEFT_IMMORTAL_ID, suffix = RIGHT_IMMORTAL_ID))
end

function make_single_model(model)
    return function (t, Xt)
        Xt_model = to_device(make_model_input([Xt]))
        tdev = to_device(Float32[t])
        return model(tdev, Xt_model)
    end
end

function prepare_batch(ts, Xts, x1s)
    guide_tgt = Flowfusion.Guide(P, ts, Xts, x1s)
    Xt_model = make_model_input(Xts)
    return (
        ts = to_device(ts),
        Xt = to_device(Xt_model),
        guide = to_device(guide_tgt),
    )
end

function batch_loss(model, batch)
    preds = model(batch.ts, batch.Xt)
    return Flowfusion.floss(P, batch.Xt, preds, batch.guide, Flowfusion.scalefloss(P, batch.ts, 1))
end

function eval_loss(model, rng, pairs; batches = 8)
    losses = Float32[]
    for _ in 1:batches
        ts = rand(rng, Float32, length(pairs))
        Xts = [Flowfusion.bridge(P, first(pairs[i]), last(pairs[i]), ts[i]) for i in eachindex(pairs)]
        batch = prepare_batch(ts, Xts, last.(pairs))
        loss = batch_loss(model, batch)
        push!(losses, Float32(loss))
    end
    return mean(losses)
end

epochs = parse(Int, get(ENV, "MODPIP_EPOCHS", "4"))
steps_per_epoch = parse(Int, get(ENV, "MODPIP_STEPS", "100"))
batch_size = parse(Int, get(ENV, "MODPIP_BATCH", "8"))
lr = parse(Float64, get(ENV, "MODPIP_LR", "1e-3"))
seed = parse(Int, get(ENV, "MODPIP_SEED", "0"))

rng = MersenneTwister(seed)
train_rng = MersenneTwister(seed + 1)
val_rng = MersenneTwister(seed + 2)
demo_rng = MersenneTwister(seed + 3)

val_pairs = [sample_pair(val_rng) for _ in 1:16]
demo_x0s = [DiscreteState(KPROC, sample_x0(demo_rng)) for _ in 1:3]

model = ModPIPModel(; d = 128, depth = 4, max_len = 192, K = KMODEL)
model = to_device(model)
opt_state = Flux.setup(Optimisers.Adam(lr), model)

@info "modpip setup" device=device_name() epochs batch_size steps_per_epoch lr x0_prior=X0_PRIOR

for epoch in 1:epochs
    losses = Float32[]
    for step_idx in 1:steps_per_epoch
        xpairs = [sample_pair(train_rng) for _ in 1:batch_size]
        x0s = first.(xpairs)
        x1s = last.(xpairs)
        ts = rand(train_rng, Float32, batch_size)
        Xts = [Flowfusion.bridge(P, x0s[b], x1s[b], ts[b]) for b in 1:batch_size]
        batch = prepare_batch(ts, Xts, x1s)

        loss, grads = Flux.withgradient(model) do m
            batch_loss(m, batch)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, Float32(loss))

        if step_idx % 25 == 0 || step_idx == 1
            @info "modpip train" epoch step=step_idx loss=Float32(loss)
        end
    end

    vloss = eval_loss(model, val_rng, val_pairs)
    sampler = make_single_model(model)
    samples = [encode(tensor(Flowfusion.gen(P, x0, sampler, 0f0:0.02f0:1f0))) for x0 in demo_x0s]
    @info "modpip epoch" epoch train_loss=mean(losses) val_loss=vloss samples=samples
end
