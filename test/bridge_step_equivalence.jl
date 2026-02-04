using Flowfusion
using ForwardBackward
using Manifolds
using Random
using Test
using Distributions

const T = Float32
Random.seed!(20260203)

abstract type BridgeDomain end
struct ContinuousDomain <: BridgeDomain
    dim::Int
    n_samples::Int
end
struct DiscreteDomain <: BridgeDomain
    K::Int
    sites::Int
    n_samples::Int
end
struct ManifoldDomain <: BridgeDomain
    M::AbstractManifold
    n_samples::Int
end

struct BridgeCase{P<:Process,D<:BridgeDomain}
    name::String
    process::P
    domain::D
    supports_three_time::Bool
end

# -------------------------
# Sampling helpers
# -------------------------
function sample_pair(d::ContinuousDomain, rng)
    X0 = ContinuousState(randn(rng, T, d.dim, d.n_samples))
    X1 = ContinuousState(randn(rng, T, d.dim, d.n_samples) .+ T(2))
    return X0, X1
end

function sample_categorical(rng, probs::AbstractVector{<:Real}, dims...)
    cdf = cumsum(probs)
    out = Array{Int}(undef, dims...)
    for i in eachindex(out)
        u = rand(rng)
        out[i] = searchsortedfirst(cdf, u)
    end
    return out
end

function sample_pair(d::DiscreteDomain, rng)
    K = d.K
    # Two different categorical distributions
    p0 = T.([0.55, 0.2, 0.15, 0.07, 0.03][1:K])
    p1 = T.([0.05, 0.15, 0.2, 0.25, 0.35][1:K])
    p0 ./= sum(p0)
    p1 ./= sum(p1)
    X0 = DiscreteState(K, sample_categorical(rng, p0, d.sites, d.n_samples))
    X1 = DiscreteState(K, sample_categorical(rng, p1, d.sites, d.n_samples))
    return X0, X1
end

function sample_pair(d::ManifoldDomain, rng)
    M = d.M
    # Use two different base points and perturb around each
    p0 = rand(M)
    p1 = rand(M)
    x0 = [ForwardBackward.perturb(M, p0, 0.05) for _ in 1:d.n_samples]
    x1 = [ForwardBackward.perturb(M, p1, 0.05) for _ in 1:d.n_samples]
    return ManifoldState(M, x0), ManifoldState(M, x1)
end

# -------------------------
# Distribution comparison
# -------------------------
function flatten_samples(X)
    A = tensor(X)
    n = size(A, ndims(A))
    return reshape(A, :, n)
end

function ks_statistic(a::AbstractVector, b::AbstractVector)
    as = sort(a)
    bs = sort(b)
    na = length(as)
    nb = length(bs)
    ia = 1
    ib = 1
    cdf_a = 0.0
    cdf_b = 0.0
    d = 0.0
    while ia <= na || ib <= nb
        if ib > nb || (ia <= na && as[ia] <= bs[ib])
            v = as[ia]
            while ia <= na && as[ia] == v
                ia += 1
            end
            cdf_a = (ia - 1) / na
        else
            v = bs[ib]
            while ib <= nb && bs[ib] == v
                ib += 1
            end
            cdf_b = (ib - 1) / nb
        end
        d = max(d, abs(cdf_a - cdf_b))
    end
    return d
end

function ks_pvalue(d::Real, n::Int, m::Int)
    en = sqrt(n * m / (n + m))
    λ = (en + 0.12 + 0.11 / en) * d
    s = 0.0
    for j in 1:200
        term = (-1)^(j - 1) * exp(-2 * (λ^2) * (j^2))
        s += term
        if abs(term) < 1e-10
            break
        end
    end
    p = 2 * s
    return clamp(p, 0.0, 1.0)
end

function projection_ks(Xa, Xb; rng, proj_count = 3)
    A = flatten_samples(Xa)
    B = flatten_samples(Xb)
    n = size(A, 2)
    m = size(B, 2)
    dmax = 0.0
    for _ in 1:proj_count
        v = randn(rng, size(A, 1))
        pa = vec(v' * A)
        pb = vec(v' * B)
        dmax = max(dmax, ks_statistic(pa, pb))
    end
    return dmax, ks_pvalue(dmax, n, m)
end

function manifold_distance_ks(M, X0::ManifoldState, X1::ManifoldState, Xa::ManifoldState, Xb::ManifoldState)
    n = length(Xa.state)
    m = length(Xb.state)
    d0a = Vector{Float64}(undef, n)
    d0b = Vector{Float64}(undef, m)
    d1a = Vector{Float64}(undef, n)
    d1b = Vector{Float64}(undef, m)
    @inbounds for i in eachindex(Xa.state)
        d0a[i] = distance(M, X0.state[i], Xa.state[i])
        d1a[i] = distance(M, X1.state[i], Xa.state[i])
    end
    @inbounds for i in eachindex(Xb.state)
        d0b[i] = distance(M, X0.state[i], Xb.state[i])
        d1b[i] = distance(M, X1.state[i], Xb.state[i])
    end
    ks0 = ks_statistic(d0a, d0b)
    ks1 = ks_statistic(d1a, d1b)
    dmax = max(ks0, ks1)
    return dmax, ks_pvalue(dmax, n, m)
end

function discrete_tv_distance(Xa::DiscreteState, Xb::DiscreteState)
    K = Xa.K
    counts_a = zeros(Float64, K)
    counts_b = zeros(Float64, K)
    for v in Xa.state
        counts_a[v] += 1
    end
    for v in Xb.state
        counts_b[v] += 1
    end
    pa = counts_a ./ sum(counts_a)
    pb = counts_b ./ sum(counts_b)
    return 0.5 * sum(abs.(pa .- pb))
end

function discrete_pvalue(Xa::DiscreteState, Xb::DiscreteState)
    K = Xa.K
    counts_a = zeros(Float64, K)
    counts_b = zeros(Float64, K)
    for v in Xa.state
        counts_a[v] += 1
    end
    for v in Xb.state
        counts_b[v] += 1
    end
    na = sum(counts_a)
    nb = sum(counts_b)
    total = counts_a .+ counts_b
    epsv = 1e-12
    exp_a = total .* (na / (na + nb)) .+ epsv
    exp_b = total .* (nb / (na + nb)) .+ epsv
    stat = sum(((counts_a .- exp_a) .^ 2) ./ exp_a) + sum(((counts_b .- exp_b) .^ 2) ./ exp_b)
    df = max(K - 1, 1)
    return ccdf(Chisq(df), stat)
end

function ks_threshold(case::BridgeCase; two_time::Bool=false)
    if case.domain isa ManifoldDomain
        return two_time ? 0.40 : 0.20
    end
    return 0.15
end

function dt_for_case(case::BridgeCase; two_time::Bool=false)
    if case.domain isa ManifoldDomain
        return two_time ? T(0.01) : T(0.02)
    end
    return T(0.02)
end

# -------------------------
# Bridge vs step runner
# -------------------------
step_hat(::Flowfusion.ConvexInterpolatingDiscreteFlow, X1) = onehot(X1)
function onehot_logits(X1::DiscreteState; hi = T(10), lo = T(-10))
    oh = onehot(X1)
    return lo .+ (hi - lo) .* T.(tensor(oh))
end
step_hat(::Flowfusion.DistInterpolatingDiscreteFlow, X1::DiscreteState) = onehot_logits(X1)
step_hat(::Flowfusion.DistNoisyInterpolatingDiscreteFlow, X1::DiscreteState) = onehot_logits(X1)
step_hat(::Process, X1) = X1

function step_path(P, X0, X1, steps, t_indices)
    Xt = copy(X0)
    results = Dict{Int, typeof(X0)}()
    for i in 1:length(steps)-1
        s1 = steps[i]
        s2 = steps[i + 1]
        Xt = Flowfusion.step(P, Xt, step_hat(P, X1), s1, s2)
        if i + 1 in t_indices
            results[i + 1] = copy(Xt)
        end
    end
    return results
end

function bridge_samples(P, X0, X1, t)
    return bridge(P, X0, X1, t)
end

function bridge_samples(P, X0, X1, t0, t)
    return bridge(P, X0, X1, t0, t)
end

function compare_distributions(case::BridgeCase, t_values)
    rng = MersenneTwister(20260203)
    X0, X1 = sample_pair(case.domain, rng)

    dt = dt_for_case(case; two_time=false)
    steps = collect(range(T(0), T(1), step = dt))
    step_index = Dict(round(Int, steps[i] / dt) => i for i in eachindex(steps))
    t_indices = [step_index[round(Int, t / dt)] for t in t_values]

    step_results = step_path(case.process, X0, X1, steps, Set(t_indices))

    for t in t_values
        idx = step_index[round(Int, t / dt)]
        Xt_bridge = bridge_samples(case.process, X0, X1, t)
        Xt_step = step_results[idx]

        if case.domain isa DiscreteDomain
            tv = discrete_tv_distance(Xt_bridge, Xt_step)
            pval = discrete_pvalue(Xt_bridge, Xt_step)
            @info "bridge vs step" process = case.name t = t metric = "tv" value = tv p_value = pval
            @test tv <= 0.12
        elseif case.domain isa ManifoldDomain
            ks, pval = manifold_distance_ks(case.domain.M, X0, X1, Xt_bridge, Xt_step)
            @info "bridge vs step" process = case.name t = t metric = "ks" value = ks p_value = pval
            @test ks <= ks_threshold(case; two_time=false)
        else
            ks, pval = projection_ks(Xt_bridge, Xt_step; rng = rng)
            @info "bridge vs step" process = case.name t = t metric = "ks" value = ks p_value = pval
            @test ks <= ks_threshold(case; two_time=false)
        end
    end
end

function compare_distributions_two_time(case::BridgeCase, t0, t_values)
    rng = MersenneTwister(20260203)
    X0, X1 = sample_pair(case.domain, rng)

    dt = dt_for_case(case; two_time=true)
    steps = collect(range(T(t0), T(1), step = dt))
    step_index = Dict(round(Int, (steps[i] - t0) / dt) => i for i in eachindex(steps))
    t_indices = [step_index[round(Int, (t - t0) / dt)] for t in t_values]

    step_results = step_path(case.process, X0, X1, steps, Set(t_indices))

    for t in t_values
        idx = step_index[round(Int, (t - t0) / dt)]
        Xt_bridge = bridge_samples(case.process, X0, X1, t0, t)
        Xt_step = step_results[idx]

        if case.domain isa DiscreteDomain
            tv = discrete_tv_distance(Xt_bridge, Xt_step)
            pval = discrete_pvalue(Xt_bridge, Xt_step)
            @info "bridge vs step (two-time)" process = case.name t0 = t0 t = t metric = "tv" value = tv p_value = pval
            @test tv <= 0.12
        elseif case.domain isa ManifoldDomain
            ks, pval = manifold_distance_ks(case.domain.M, X0, X1, Xt_bridge, Xt_step)
            @info "bridge vs step (two-time)" process = case.name t0 = t0 t = t metric = "ks" value = ks p_value = pval
            @test ks <= ks_threshold(case; two_time=true)
        else
            ks, pval = projection_ks(Xt_bridge, Xt_step; rng = rng)
            @info "bridge vs step (two-time)" process = case.name t0 = t0 t = t metric = "ks" value = ks p_value = pval
            @test ks <= ks_threshold(case; two_time=true)
        end
    end
end

# -------------------------
# Test battery
# -------------------------
@testset "Bridge vs step distributions" begin
    t_values = T.([0.02, 0.2, 0.5, 0.8, 0.98])
    t0 = T(0.75)
    t_values_two_time = T.([0.77, 0.85, 0.93, 0.99])

    cases = BridgeCase[
        BridgeCase("deterministic", Deterministic(), ContinuousDomain(2, 1024), true),
        BridgeCase("brownian", BrownianMotion(T(0.15)), ContinuousDomain(2, 1024), true),
        BridgeCase("ornstein_uhlenbeck", OrnsteinUhlenbeck(T(0), T(0.4), T(1.2)), ContinuousDomain(2, 1024), true),
        BridgeCase("ou_expvar", OrnsteinUhlenbeckExpVar(T(0), T(1.0), T(0.6), T(0.25); dec = T(-0.2)), ContinuousDomain(2, 1024), true),
        BridgeCase("ou_flow", OUFlow(T(1.0), T(0.6), T(0.2), T(-0.1)), ContinuousDomain(2, 1024), true),
        BridgeCase("interpolating_discrete", InterpolatingDiscreteFlow(), DiscreteDomain(5, 3, 1200), false),
        BridgeCase("noisy_interpolating_discrete", NoisyInterpolatingDiscreteFlow(T(0.2)), DiscreteDomain(5, 3, 1200), false),
        BridgeCase("dist_interpolating_discrete", DistInterpolatingDiscreteFlow(Beta(2.0, 2.0)), DiscreteDomain(5, 3, 1200), true),
        BridgeCase("dist_noisy_interpolating_discrete", DistNoisyInterpolatingDiscreteFlow(D1=Beta(2.0, 2.0), D2=Beta(2.0, 5.0), ωu=0.2), DiscreteDomain(5, 3, 1200), true),
        BridgeCase("uniform_discrete", UniformDiscrete(T(1.0)), DiscreteDomain(5, 3, 1200), true),
        BridgeCase("piq", PiQ(T(1.0), T.([0.1, 0.2, 0.3, 0.4, 0.2])), DiscreteDomain(5, 3, 1200), true),
        BridgeCase("general_discrete", GeneralDiscrete([T(-1.0) T(0.4) T(0.3) T(0.2) T(0.1);
                                                      T(0.3)  T(-1.0) T(0.4) T(0.2) T(0.1);
                                                      T(0.2)  T(0.3) T(-1.0) T(0.4) T(0.1);
                                                      T(0.1)  T(0.2) T(0.3) T(-1.0) T(0.4);
                                                      T(0.4)  T(0.1) T(0.2) T(0.3) T(-1.0)]), DiscreteDomain(5, 3, 1200), true),
        BridgeCase("manifold_torus", ManifoldProcess(T(0.2)), ManifoldDomain(Torus(2), 256), true),
        BridgeCase("manifold_so3", ManifoldProcess(T(0.2)), ManifoldDomain(SpecialOrthogonal(3), 128), true),
        BridgeCase("manifold_torus_ouexpvar", ManifoldProcess(OUBridgeExpVar(T(1.0), T(1.5), T(1e-9); dec = T(-3.0))), ManifoldDomain(Torus(2), 256), true),
        BridgeCase("manifold_so3_ouexpvar", ManifoldProcess(OUBridgeExpVar(T(1.0), T(1.5), T(1e-9); dec = T(-3.0))), ManifoldDomain(SpecialOrthogonal(3), 128), true),
    ]

    for case in cases
        @testset "$(case.name)" begin
            compare_distributions(case, t_values)
            if case.supports_three_time
                compare_distributions_two_time(case, t0, t_values_two_time)
            end
        end
    end
end
