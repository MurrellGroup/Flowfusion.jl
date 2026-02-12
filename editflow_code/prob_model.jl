# ===================== Target: ProfileMixtureEOS from examples/abs.txt =====================
# 20 AA alphabet (no '#', no '-')
const AA20      = collect("ACDEFGHIKLMNPQRSTVWY")
const TOK2ID_AA = Dict{Char,Int}(c => i for (i,c) in enumerate(AA20))  # 1..20
const K_AA = length(AA20)

# encode and truncate to a random length in 10:20 (inclusive), reproducible via `rng`
function encode_line_AA(line::AbstractString; rng::AbstractRNG, trunc_range::UnitRange{Int}=10:20)
    maxlen = rand(rng, trunc_range)
    s = uppercase(strip(line))
    out = Int[]
    sizehint!(out, min(length(s), maxlen))
    for ch in s
        if haskey(TOK2ID_AA, ch)
            push!(out, TOK2ID_AA[ch])
            if length(out) == maxlen; break; end
        end
    end
    return out
end

struct ProfileMixtureEOS
    K::Int
    parents::Vector{Vector{Int}}
    weights::Vector{Float64}
    base_token_probs::Vector{Vector{Vector{Float64}}}  # per m, per t
    L::Vector{Int}
    background::Vector{Float64}
    mode::Symbol
    h_tail::Float64
    h_inside::Float64   # end hazard for t ≤ L_m
    alpha::Float64
    beta::Float64
end

function ProfileMixtureEOS(parents::Vector{Vector{Int}};
                           K::Union{Int,Nothing}=nothing,
                           weights::AbstractVector=Float64[],
                           α::Real=1.0, β::Real=20.0,
                           background::Union{AbstractVector{<:Real},Nothing}=nothing,
                           mode::Symbol=:soft, h_tail::Real=0.99, h_inside::Real=0.001)
    K === nothing && (K = maximum([isempty(p) ? 1 : maximum(p) for p in parents]))
    @assert all(all(1 .<= p .<= K) for p in parents)
    @assert mode in (:hard, :soft)
    if mode === :soft
        @assert 0.0 < h_tail   < 1.0
        @assert 0.0 < h_inside < 1.0
    end
    N = length(parents)
    weights    = isempty(weights) ? fill(1.0/N, N) : collect(weights ./ sum(weights))
    background = background === nothing ? fill(1.0/K, K) : collect(background ./ sum(background))
    base_token_probs = Vector{Vector{Vector{Float64}}}(undef, N)
    L = Int[]
    for (m, seq) in enumerate(parents)
        push!(L, length(seq))
        prof = Vector{Vector{Float64}}(undef, length(seq))
        for t in 1:length(seq)
            λv = α .* background
            λv[seq[t]] += β
            prof[t] = λv ./ sum(λv)
        end
        base_token_probs[m] = prof
    end
    ProfileMixtureEOS(K, parents, weights, base_token_probs, L, background, mode,
                      float(h_tail), float(h_inside), float(α), float(β))
end

@inline function _hazard(M::ProfileMixtureEOS, m::Int, t::Int)
    Lm = M.L[m]
    if M.mode === :hard
        return t <= Lm ? 0.0 : 1.0
    else
        return t <= Lm ? M.h_inside : M.h_tail
    end
end

@inline function _logsumexp(v::AbstractVector{<:Real})
    m = maximum(v); !isfinite(m) && return m
    s = 0.0
    @inbounds for i in eachindex(v); s += exp(v[i] - m); end
    return m + log(s)
end

function _logpmf_component(M::ProfileMixtureEOS, m::Int, x::Vector{Int})
    n = length(x)
    if n == 0
        qend = _hazard(M, m, 1)
        return qend > 0.0 ? log(qend) : -Inf
    end
    lp = 0.0
    @inbounds for t in 1:n
        s = 1.0 - _hazard(M, m, t)
        p = (t <= M.L[m]) ? M.base_token_probs[m][t][x[t]] : M.background[x[t]]
        q = s * p
        if q <= 0.0; return -Inf; end
        lp += log(q)
    end
    qend = _hazard(M, m, n + 1)
    return qend > 0.0 ? lp + log(qend) : -Inf
end

function logpmf(M::ProfileMixtureEOS, x::Vector{Int})
    @assert all(1 .<= x .<= M.K) "PMF expects tokens in 1..K_AA"
    tmp = similar(M.weights)
    @inbounds for m in 1:length(M.parents)
        tmp[m] = log(M.weights[m]) + _logpmf_component(M, m, x)
    end
    return _logsumexp(tmp)
end
pmf(M::ProfileMixtureEOS, x::Vector{Int}) = exp(logpmf(M, x))

@inline function _catdraw(rng::AbstractRNG, p::AbstractVector{<:Real})
    r = rand(rng); acc = 0.0
    @inbounds for i in eachindex(p)
        acc += p[i]
        if r <= acc; return i; end
    end
    return lastindex(p)
end

function _sample_from_component(M::ProfileMixtureEOS, m::Int; rng::AbstractRNG)
    seq = Int[]; t = 1
    while true
        h = _hazard(M, m, t)
        if h > 0.0 && rand(rng) < h
            return seq
        end
        a = (t <= M.L[m]) ? _catdraw(rng, M.base_token_probs[m][t]) : _catdraw(rng, M.background)
        push!(seq, a); t += 1
    end
end

function sample(M::ProfileMixtureEOS; rng::AbstractRNG=Random.default_rng(), return_component::Bool=false)
    m = _catdraw(rng, M.weights)
    seq = _sample_from_component(M, m; rng=rng)
    return return_component ? (seq, m) : seq
end
import Random: rand
rand(rng::AbstractRNG, M::ProfileMixtureEOS) = sample(M; rng=rng)
rand(M::ProfileMixtureEOS) = sample(M)

function build_target_from_abs(; file::AbstractString=joinpath(@__DIR__, "abs.txt"),
                               maxlines::Int=100,
                               seed::Int=0,
                               α=1.0, β=20.0, mode::Symbol=:soft,
                               h_tail=0.99, h_inside=0.001,
                               trunc_range::UnitRange{Int}=10:20)
    @assert isfile(file) "abs.txt not found at $file"
    rng = Random.MersenneTwister(seed)
    parents = Vector{Vector{Int}}()
    open(file, "r") do io
        for (i, line) in enumerate(eachline(io))
            i > maxlines && break
            seq = encode_line_AA(line; rng=rng, trunc_range=trunc_range)
            if !isempty(seq); push!(parents, seq); end
        end
    end
    @assert !isempty(parents) "No non-empty sequences in first $maxlines lines after filtering."
    ProfileMixtureEOS(parents; K=K_AA, α=α, β=β, mode=mode, h_tail=h_tail, h_inside=h_inside)
end

const PM = build_target_from_abs(file=joinpath(@__DIR__, "abs.txt"))


