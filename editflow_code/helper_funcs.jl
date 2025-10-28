using StringDistances, Statistics, ProgressMeter, Plots, StatsBase


# Sampling helpers
function state_to_PM_string(P::FF.EditFlow, st::FF.DiscreteState)
    xs = FF.tensor(st)
    pad = hasfield(typeof(P), :padding_token) ? P.padding_token : -1
    io = IOBuffer()
    @inbounds for tok in xs
        if tok == 0
            write(io, '>')
        elseif tok == pad
            continue
        else
            write(io, AA20[tok])
        end
    end
    return String(take!(io))
end

states_to_PM_strings(P::FF.EditFlow, sts::Vector{<:FF.DiscreteState}) =
    [state_to_PM_string(P, s) for s in sts]

function sample_gen_n(P::FF.EditFlow, model; n::Int=10, ts=0f0:0.01f0:1f0, rng=Random.default_rng())
    gens = Vector{Any}(undef, n)
    for i in 1:n
        x0s, _, _ = make_minibatch(1, P; rng=rng)
        bos = P.bos_token
        toks = collect(FF.tensor(x0s[1]))
        x0_bos = FF.DiscreteState(P.k, vcat([bos], toks))
        gens[i] = FF.gen(P, x0_bos, model, ts)
    end
    return gens
end

function sample_gen_10_strings(P::FF.EditFlow, model; ts=0f0:0.01f0:1f0)
    outs = sample_gen_n(P, model; n=20, ts=ts)
    return map(outs) do o
        if o isa FF.DiscreteState
            state_to_PM_string(P, o)
        elseif o isa AbstractVector{<:FF.DiscreteState}
            states_to_PM_strings(P, o)
        else
            o
        end
    end
end

final_state_from_gen(res) = res isa FF.DiscreteState ? res : (res isa AbstractVector{<:FF.DiscreteState} ? last(res) : res)

function to_aa_tokens(P::FF.EditFlow, st::FF.DiscreteState)
    xs = FF.tensor(st)
    pad = hasfield(typeof(P), :padding_token) ? P.padding_token : -1
    toks = Vector{Int}()
    @inbounds for tok in xs
        if tok == 0 || tok == pad
            continue
        else
            push!(toks, Int(tok))
        end
    end
    return toks
end

function min_lev_to_set(queries::Vector{String}, refs::Vector{String}; normalize::Bool=true)
    d = Levenshtein()
    ref_lens = length.(refs)
    out = Vector{Float64}(undef, length(queries))
    @showprogress for (i, q) in enumerate(queries)
        Lq = length(q)
        best = typemax(Int); best_Lr = 1
        for (r, Lr) in zip(refs, ref_lens)
            lb = abs(Lq - Lr)
            lb >= best && continue
            dist = evaluate(d, q, r)
            if dist < best
                best = dist; best_Lr = Lr
                best == 0 && break
            end
        end
        out[i] = normalize ? best / max(Lq, best_Lr) : best
    end
    return out
end

function plot_lev_dist(name, val_seqs, gen_seqs, train_seqs; title="Novelty vs. training set", max_seqs=1000)
    sample_if_large(seqs, n=max_seqs) = length(seqs) > n ? rand(seqs, n) : seqs

    val_sample = sample_if_large(val_seqs)
    gen_sample = sample_if_large(gen_seqs)
    train_sample = sample_if_large(train_seqs)

    val_lev = min_lev_to_set(val_sample, train_sample; normalize=true)
    gen_lev = min_lev_to_set(gen_sample, train_sample; normalize=true)

    # Robust bins even when all values are identical
    minv = minimum([minimum(val_lev), minimum(gen_lev)])
    maxv = maximum([maximum(val_lev), maximum(gen_lev)])
    bins = (minv == maxv) ? range(0.0, 1.0, length=25) : range(minv, maxv, length=25)
    p = histogram(val_lev; normalize=:probability, alpha=0.5, label="Natural", 
                  bins=bins, xlabel="Min normalized Levenshtein", ylabel="Density", title=title)
    histogram!(gen_lev; normalize=:probability, alpha=0.5, label="Generated", bins=bins)
    
    #println("val_lev: ", val_lev)
    #println("gen_lev: ", gen_lev)

    #path = joinpath(@__DIR__, "figures", "$(name).pdf")
    #mkpath(dirname(path))
    #savefig(p, path)
    #savefig(p, "/figures/$(name).pdf")
    return p
end

function AA_counts(seqs)
    counts = countmap(Iterators.flatten(seqs))
    labels = sort!(collect(keys(counts)))
    return labels, counts
end

function plot_AA_dist(name, real_seqs, gen_seqs; title="Normalized AA frequencies")
    # Always show the full alphabet and filter invalid chars
    labels = collect(AA20)
    alphabet = Set(labels)
    r_counts = countmap(Iterators.flatten((ch for s in real_seqs for ch in s if ch in alphabet)))
    g_counts = countmap(Iterators.flatten((ch for s in gen_seqs  for ch in s if ch in alphabet)))
    r = Float64[get(r_counts, c, 0) for c in labels]
    g = Float64[get(g_counts, c, 0) for c in labels]
    rs = sum(r); r = rs > 0 ? r ./ rs : fill(0.0, length(r))
    gs = sum(g); g = gs > 0 ? g ./ gs : fill(0.0, length(g))
    # Grouped bars without unsupported bar_position
    p = bar(string.(labels), [r g];
            label=["Natural" "Generated"], xlabel="AA", ylabel="Proportion",
            title=title, alpha=0.5)
    return p
end

function len_dist(seqs; fig_name="len_dist.pdf", sort_labels=true)
    isempty(seqs) && return (Int[], Float64[])
    # Count only AA20 letters (ignore BOS '>' and any non-AA chars)
    alphabet = Set(AA20)
    lens = [count(ch -> ch in alphabet, s) for s in seqs]
    cm   = countmap(lens)                 # how many seqs of each length
    Ls   = collect(keys(cm))
    sort_labels && sort!(Ls)
    props = [cm[L] for L in Ls] ./ max(length(lens), 1)
    return Ls, props
end
function plot_len_dist(name, real_seqs, gen_seqs; title="Sequence length distribution")
    real_labels, real_props = len_dist(real_seqs)
    gen_labels, gen_props   = len_dist(gen_seqs)

    # Build unified x-axis and zero-fill missing bins
    all_labels = sort!(union(real_labels, gen_labels))
    rmap = Dict(real_labels .=> real_props)
    gmap = Dict(gen_labels .=> gen_props)
    r = [get(rmap, L, 0.0) for L in all_labels]
    g = [get(gmap, L, 0.0) for L in all_labels]

    p = bar(string.(all_labels), [r g];
            label=["Natural" "Generated"], xlabel="Length", ylabel="Proportion",
            title=title, alpha=0.5)
    #savefig(p, "../img/unconditional/len_dist_$(name).pdf")
    return p
end