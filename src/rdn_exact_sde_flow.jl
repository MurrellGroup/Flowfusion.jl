# Exact endpoint-conditioned SDE process matching the La-Proteina RDN sampler
# in the non-annealed regime. The bridge works in the residual variable
#   z_t = (x_t - τ(t) x_1) / (1 - τ(t))
# for which the conditional dynamics become an OU process with time-varying
# decay rate λ(t) = g(t) / (1 - t)^2.

using ForwardBackward

struct RDNExactSDEFlow{T<:Real,U} <: Process
    dim::Int
    zero_com::Bool
    schedule::Symbol
    schedule_param::T
    sde_gt_mode::Symbol
    sde_gt_param::T
    sc_scale_noise::T
    gt_eps::T
    gt_clamp::U
end

function RDNExactSDEFlow(
    dim::Int;
    zero_com::Bool = false,
    schedule::Symbol = :linear,
    schedule_param::Real = 1.0f0,
    sde_gt_mode::Symbol = :const,
    sde_gt_param::Real = 1.0f0,
    sc_scale_noise::Real = 1.0f0,
    gt_eps::Real = 1.0f-2,
    gt_clamp::Union{Nothing,Real} = nothing,
)
    sc_scale_noise >= 0 || error("sc_scale_noise must be >= 0, got $sc_scale_noise")
    if !isnothing(gt_clamp)
        error("RDNExactSDEFlow does not support gt_clamp yet; exact bridge derivation assumes unclamped g(t)")
    end
    if sde_gt_mode in (:tan, Symbol("1-t/t"), Symbol("1/t")) && Float32(sde_gt_param) != 1.0f0
        error("RDNExactSDEFlow only supports sde_gt_param=1 for mode $sde_gt_mode")
    end
    clamp_value = isnothing(gt_clamp) ? nothing : Float32(gt_clamp)
    return RDNExactSDEFlow(
        dim,
        zero_com,
        schedule,
        Float32(schedule_param),
        sde_gt_mode,
        Float32(sde_gt_param),
        Float32(sc_scale_noise),
        Float32(gt_eps),
        clamp_value,
    )
end

function schedule_transform(P::RDNExactSDEFlow, u::T) where T<:Real
    p = T(P.schedule_param)
    if P.schedule == :linear
        return u
    elseif P.schedule == :power
        return u^p
    elseif P.schedule == :log
        denom = one(T) - T(10)^(-p)
        return (one(T) - T(10)^(-p * u)) / denom
    else
        error("Unknown schedule: $(P.schedule)")
    end
end

function schedule_transform(P::RDNExactSDEFlow, u::AbstractArray{T}) where T<:Real
    p = T(P.schedule_param)
    if P.schedule == :linear
        return u
    elseif P.schedule == :power
        return u .^ p
    elseif P.schedule == :log
        denom = one(T) - T(10)^(-p)
        return (one(T) .- T(10) .^ (-p .* u)) ./ denom
    else
        error("Unknown schedule: $(P.schedule)")
    end
end

function sample_rdn_noise(P::RDNExactSDEFlow, shape...; T=Float32, mask=nothing)
    d = P.dim
    noise = randn(T, d, shape...)

    if P.zero_com
        noise = _force_zero_com(noise, mask)
    elseif !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        noise = noise .* mask_exp
    end

    return noise
end

function _rdn_exact_gt(P::RDNExactSDEFlow, t::T) where T<:Real
    t_clamped = clamp(t, T(0), T(1) - T(1e-8))
    eps = T(P.gt_eps)

    gt = if P.sde_gt_mode == :const
        T(P.sde_gt_param)
    elseif P.sde_gt_mode == :linear
        T(P.sde_gt_param) * t_clamped
    elseif P.sde_gt_mode == :tan
        num = sin((one(T) - t_clamped) * T(π / 2))
        den = cos((one(T) - t_clamped) * T(π / 2)) + eps
        T(π / 2) * num / den
    elseif P.sde_gt_mode == Symbol("1-t/t")
        (one(T) - t_clamped) / (t_clamped + eps)
    elseif P.sde_gt_mode == Symbol("1/t")
        one(T) / (t_clamped + eps)
    else
        error("Unsupported sde_gt_mode $(P.sde_gt_mode)")
    end

    if !isnothing(P.gt_clamp)
        gt = clamp(gt, zero(T), T(P.gt_clamp))
    end

    return gt
end

function _rdn_simpson(f, a::Float64, b::Float64, fa::Float64, fm::Float64, fb::Float64)
    return (b - a) * (fa + 4fm + fb) / 6
end

function _rdn_adaptive_simpson(
    f,
    a::Float64,
    b::Float64,
    fa::Float64,
    fm::Float64,
    fb::Float64,
    whole::Float64,
    tol::Float64,
    depth::Int,
)
    m = (a + b) / 2
    lm = (a + m) / 2
    rm = (m + b) / 2
    flm = f(lm)
    frm = f(rm)
    left = _rdn_simpson(f, a, m, fa, flm, fm)
    right = _rdn_simpson(f, m, b, fm, frm, fb)
    delta = left + right - whole
    if depth <= 0 || abs(delta) <= 15 * tol
        return left + right + delta / 15
    end
    return _rdn_adaptive_simpson(f, a, m, fa, flm, fm, left, tol / 2, depth - 1) +
           _rdn_adaptive_simpson(f, m, b, fm, frm, fb, right, tol / 2, depth - 1)
end

function _rdn_adaptive_simpson(f, a::Float64, b::Float64; tol::Float64 = 1e-10, maxdepth::Int = 20)
    if !(b > a)
        return 0.0
    end
    fa = f(a)
    fb = f(b)
    m = (a + b) / 2
    fm = f(m)
    whole = _rdn_simpson(f, a, b, fa, fm, fb)
    return _rdn_adaptive_simpson(f, a, b, fa, fm, fb, whole, tol, maxdepth)
end

function _rdn_lambda_integral_tan(P::RDNExactSDEFlow, tau0::T, tau1::T) where T<:Real
    tau1 <= tau0 && return zero(Float64)
    q0 = 1.0 - Float64(tau0)
    q1 = 1.0 - Float64(tau1)
    q1 <= 0.0 && return Inf

    beta = Float64(π / 2)
    eps = Float64(P.gt_eps)
    singular_coeff = beta^2 / (1.0 + eps)

    regular(q) = begin
        if q <= 1e-8
            return 0.0
        end
        numer = beta * sin(beta * q)
        denom = (cos(beta * q) + eps) * q^2
        return numer / denom - singular_coeff / q
    end

    return singular_coeff * log(q0 / q1) + _rdn_adaptive_simpson(regular, q1, q0)
end

function _rdn_lambda_integral(P::RDNExactSDEFlow, tau0::T, tau1::T) where T<:Real
    tau1 <= tau0 && return zero(Float64)
    tau1 >= one(T) - T(1e-12) && return Inf
    e = Float64(P.gt_eps)
    s = Float64(tau0)
    t = Float64(tau1)

    if P.sde_gt_mode == :const
        c = Float64(P.sde_gt_param)
        return c * (1 / (1 - t) - 1 / (1 - s))
    elseif P.sde_gt_mode == :linear
        c = Float64(P.sde_gt_param)
        return c * ((1 / (1 - t) + log(1 - t)) - (1 / (1 - s) + log(1 - s)))
    elseif P.sde_gt_mode == Symbol("1-t/t")
        coeff = 1 / (1 + e)
        return coeff * (log((t + e) / (s + e)) - log((1 - t) / (1 - s)))
    elseif P.sde_gt_mode == Symbol("1/t")
        coeff_log = 1 / (1 + e)^2
        coeff_pole = 1 / (1 + e)
        return coeff_log * log(((t + e) * (1 - s)) / ((s + e) * (1 - t))) +
               coeff_pole * (1 / (1 - t) - 1 / (1 - s))
    elseif P.sde_gt_mode == :tan
        return _rdn_lambda_integral_tan(P, tau0, tau1)
    else
        error("Unsupported sde_gt_mode $(P.sde_gt_mode)")
    end
end

function _rdn_exact_decay(P::RDNExactSDEFlow, tau0::T, tau1::T) where T<:Real
    lambda_int = _rdn_lambda_integral(P, tau0, tau1)
    if !isfinite(lambda_int)
        return zero(T)
    end
    decay = exp(-lambda_int)
    return clamp(T(decay), zero(T), one(T))
end

function _rdn_exact_bridge_actual_time(P::RDNExactSDEFlow, Xs::ContinuousState, X1::ContinuousState, tau0, tau1)
    T = eltype(tensor(Xs))

    tau1 >= one(T) - T(1e-7) && return copy(X1)
    tau1 <= tau0 + T(1e-8) && return copy(Xs)
    tau0 >= one(T) - T(1e-7) && error("Cannot bridge forward from tau0=$tau0: no time remains")

    xs = tensor(Xs)
    x1 = tensor(X1)
    d = ndims(xs)

    tau0_exp = expand(tau0, d)
    tau1_exp = expand(tau1, d)

    z0 = (xs .- tau0_exp .* x1) ./ (one(T) .- tau0_exp)
    alpha = _rdn_exact_decay(P, T(tau0), T(tau1))

    z1 = alpha .* z0
    if P.sc_scale_noise > T(1e-8)
        eps = randn(T, size(xs))
        if P.zero_com
            eps = _force_zero_com(eps, nothing)
        end
        noise_std = sqrt(max(zero(T), T(P.sc_scale_noise) * (one(T) - alpha^2)))
        z1 .+= noise_std .* eps
    end

    x_t = tau1_exp .* x1 .+ (one(T) .- tau1_exp) .* z1
    if P.zero_com
        x_t = _force_zero_com(x_t, nothing)
    end

    return ContinuousState(x_t)
end

function ForwardBackward.endpoint_conditioned_sample(
    X0::ContinuousState,
    X1::ContinuousState,
    P::RDNExactSDEFlow,
    tF,
    tB,
)
    T = eltype(tF)
    u = tF ./ (tF .+ tB)
    tau = schedule_transform(P, T.(u))
    return _rdn_exact_bridge_actual_time(P, X0, X1, zero(T), tau)
end

function ForwardBackward.endpoint_conditioned_sample(
    X0::ContinuousState,
    X1::ContinuousState,
    P::RDNExactSDEFlow,
    t,
)
    T = eltype(t)
    tau = schedule_transform(P, T.(t))
    return _rdn_exact_bridge_actual_time(P, X0, X1, zero(T), tau)
end

function ForwardBackward.endpoint_conditioned_sample(
    X0::ContinuousState,
    X1::ContinuousState,
    P::RDNExactSDEFlow,
    t0,
    t,
    t1,
)
    T = eltype(t)
    abs(T(t1) - one(T)) <= T(1e-5) || error("RDNExactSDEFlow expects endpoint time 1, got t1=$t1")
    tau0 = schedule_transform(P, T.(t0))
    tau = schedule_transform(P, T.(t))
    return _rdn_exact_bridge_actual_time(P, X0, X1, tau0, tau)
end

function step(P::RDNExactSDEFlow, Xₜ::ContinuousState, X̂₁::ContinuousState, s₁, s₂)
    T = eltype(tensor(Xₜ))
    tau1 = schedule_transform(P, T(s₁))
    tau2 = schedule_transform(P, T(s₂))
    return _rdn_exact_bridge_actual_time(P, Xₜ, X̂₁, tau1, tau2)
end

function step(P::RDNExactSDEFlow, Xₜ::ContinuousState, X̂₁::AbstractArray, s₁, s₂)
    return step(P, Xₜ, ContinuousState(X̂₁), s₁, s₂)
end
