# RDN Flow: Flow matching on (R^d)^n
# For protein coordinates and latent spaces

using ForwardBackward

# Simple mean implementation (avoid Statistics dependency)
_mean(x; dims) = sum(x; dims=dims) ./ size(x, dims...)

"""
    RDNFlow(dim::Int; zero_com::Bool=false, schedule::Symbol=:linear, schedule_param::Real=1.0,
            sde_gt_mode::Symbol=:const, sde_gt_param::Real=1.0,
            sc_scale_noise::Real=0.0, sc_scale_score::Real=1.0, t_lim_ode::Real=1.0)

Flow matching process on (R^d)^n where n is the number of elements (e.g., residues)
and d is the dimensionality per element.

Uses scheduled interpolation for bridging: x_t = (1-τ(u))*x_0 + τ(u)*x_1
where τ(u) is the schedule transform applied to uniform progress u.

# Arguments
- `dim`: Dimensionality d (e.g., 3 for CA coordinates, 8 for latents)
- `zero_com`: Whether to enforce zero center of mass (typically true for coordinates)
- `schedule`: Time schedule mode (:linear, :power, :log). Default :linear
- `schedule_param`: Parameter for schedule (p for power/log). Default 1.0
- `sde_gt_mode`: Mode for g(t) noise schedule (:const, :tan, Symbol("1/t"), Symbol("1-t/t"))
- `sde_gt_param`: Base parameter for g(t) schedule (default 1.0)
- `sc_scale_noise`: Noise scaling factor (0 = ODE, >0 = SDE). Default 0.0
- `sc_scale_score`: Score scaling factor (default 1.0)
- `t_lim_ode`: Switch to pure ODE above this time (default 1.0 = never switch)

# Schedule modes
- `:linear` - τ(u) = u (standard linear interpolation)
- `:power` - τ(u) = u^p (slow to reach high t, used for latents)
- `:log` - τ(u) = (1 - 10^(-p*u)) / (1 - 10^(-p)) (fast to reach high t, used for CA)

# Example
```julia
# For CA coordinates (3D, zero COM, log schedule) - la-proteina defaults
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0,
               sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0,
               sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

# For local latents (8D, no zero COM, power schedule) - la-proteina defaults
P_latent = RDNFlow(8; zero_com=false, schedule=:power, schedule_param=2.0,
                   sde_gt_mode=:tan, sde_gt_param=1.0,
                   sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

# Combined product space (use tuple)
P = (P_ca, P_latent)
```
"""
struct RDNFlow{T<:Real} <: Process
    dim::Int
    zero_com::Bool
    schedule::Symbol
    schedule_param::T
    sde_gt_mode::Symbol
    sde_gt_param::T
    sc_scale_noise::T
    sc_scale_score::T
    t_lim_ode::T
end

function RDNFlow(dim::Int; zero_com::Bool=false, schedule::Symbol=:linear,
                 schedule_param::Real=1.0f0, sde_gt_mode::Symbol=:const,
                 sde_gt_param::Real=1.0f0, sc_scale_noise::Real=0.0f0,
                 sc_scale_score::Real=1.0f0, t_lim_ode::Real=1.0f0)
    RDNFlow(dim, zero_com, schedule, Float32(schedule_param), sde_gt_mode,
            Float32(sde_gt_param), Float32(sc_scale_noise), Float32(sc_scale_score),
            Float32(t_lim_ode))
end

"""
    schedule_transform(P::RDNFlow, u)

Transform uniform progress u ∈ [0,1] to actual interpolation time τ(u).

Supported schedules:
- `:linear` - τ(u) = u
- `:power` - τ(u) = u^p (stays at low t longer)
- `:log` - τ(u) = (1 - 10^(-p*u)) / (1 - 10^(-p)) (reaches high t quickly)
"""
function schedule_transform(P::RDNFlow, u::T) where T<:Real
    p = T(P.schedule_param)
    if P.schedule == :linear
        return u
    elseif P.schedule == :power
        return u^p
    elseif P.schedule == :log
        # τ(u) = (1 - 10^(-p*u)) / (1 - 10^(-p))
        denom = one(T) - T(10)^(-p)
        return (one(T) - T(10)^(-p * u)) / denom
    else
        error("Unknown schedule: $(P.schedule)")
    end
end

# Vectorized version
function schedule_transform(P::RDNFlow, u::AbstractArray{T}) where T<:Real
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

"""
    sample_rdn_noise(P::RDNFlow, shape...; T=Float32, mask=nothing)

Sample Gaussian noise from the reference distribution.
If zero_com=true, centers the noise to have zero center of mass.

# Arguments
- `P`: RDNFlow process
- `shape`: Tuple of dimensions, should be (n, batch_dims...)
- `T`: Element type
- `mask`: Optional mask [n, batch...] where true = include in COM

# Returns
- Noise array of shape [d, n, batch...]
"""
function sample_rdn_noise(P::RDNFlow, shape...; T=Float32, mask=nothing)
    d = P.dim
    noise = randn(T, d, shape...)

    if P.zero_com
        noise = _force_zero_com(noise, mask)
    elseif !isnothing(mask)
        # Just apply mask without centering
        mask_exp = reshape(mask, 1, size(mask)...)
        noise = noise .* mask_exp
    end

    return noise
end

"""
Force zero center of mass along the sequence dimension (dim 2).
"""
function _force_zero_com(x::AbstractArray{T}, mask=nothing) where T
    if isnothing(mask)
        # Simple mean subtraction along dim 2
        com = _mean(x; dims=2)
        return x .- com
    else
        # Masked mean
        mask_exp = reshape(mask, 1, size(mask)...)  # [1, n, batch...]
        x_masked = x .* mask_exp
        n_valid = max.(sum(mask_exp; dims=2), one(T))  # [1, 1, batch...]
        com = sum(x_masked; dims=2) ./ n_valid
        return (x .- com) .* mask_exp
    end
end

# Bridge implementation - scheduled interpolation
# The schedule transforms uniform progress u to actual interpolation time τ(u)

"""
    rdn_bridge(P::RDNFlow, X0::ContinuousState, X1::ContinuousState, u)

Bridge from X0 to X1 at uniform progress u ∈ [0,1].
The actual interpolation uses τ(u) where τ is the schedule transform.
Result: X_τ = (1-τ(u))*X0 + τ(u)*X1
"""
function rdn_bridge(P::RDNFlow, X0::ContinuousState, X1::ContinuousState, u)
    T = eltype(tensor(X0))
    d = ndims(X0.state)

    # Apply schedule transform to get actual interpolation time
    tau = schedule_transform(P, T.(u))

    tau_exp = expand(tau, d)
    one_minus_tau = expand(one(T) .- tau, d)

    result = X0.state .* one_minus_tau .+ X1.state .* tau_exp

    if P.zero_com && size(result, 2) > 1
        result = _force_zero_com(result, nothing)
    end

    return ContinuousState(result)
end

# Override endpoint_conditioned_sample for ForwardBackward compatibility
function ForwardBackward.endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, P::RDNFlow, tF, tB)
    T = eltype(tF)
    # tF and tB represent forward/backward times. Compute uniform progress u.
    u = tF ./ (tF .+ tB)
    return rdn_bridge(P, X0, X1, u)
end

# 3-arg version for compatibility (t is uniform progress)
function ForwardBackward.endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, P::RDNFlow, t)
    return rdn_bridge(P, X0, X1, t)
end

# 6-arg version needed by Flowfusion.bridge
function ForwardBackward.endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, P::RDNFlow, t0, t, t1)
    # t0 is start time (usually 0), t is current time, t1 is end time (usually 1)
    # Compute progress as (t - t0) / (t1 - t0)
    T = eltype(t)
    u = (t .- t0) ./ (t1 .- t0 .+ T(1e-8))
    return rdn_bridge(P, X0, X1, u)
end

# Loss scaling for RDNFlow - same as other continuous processes
# Uses scalefloss from Flowfusion with pow=2 to get 1/(1-t)^2
# This is already handled by the default scalefloss function

# SDE step for inference (optional, for stochastic sampling)
"""
    rdn_sde_step(P::RDNFlow, x_t, v, t, dt, mask=nothing)

Single SDE integration step using:
dx_t = v dt + sqrt(2 * g(t)) dW_t

where g(t) is determined by P.sde_gt_mode and P.sde_gt_param.
"""
function rdn_sde_step(P::RDNFlow, x_t::AbstractArray{T}, v::AbstractArray{T}, t::T, dt::T, mask=nothing) where T
    gt = _compute_gt(P, t)

    # Deterministic component
    delta_x = v .* dt

    # Stochastic component if gt > 0
    if gt > 0
        noise = randn(T, size(x_t))
        if P.zero_com
            noise = _force_zero_com(noise, mask)
        end
        std = sqrt(T(2) * gt * dt)
        delta_x = delta_x .+ std .* noise
    end

    x_next = x_t .+ delta_x

    # Apply mask and possibly re-center
    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        x_next = x_next .* mask_exp
    end
    if P.zero_com
        x_next = _force_zero_com(x_next, mask)
    end

    return x_next
end

"""
    _compute_gt(P::RDNFlow, t; clamp_val=1e5, eps=1e-2)

Compute g(t) noise schedule value at time t.

Supported modes:
- `:const` - constant: g(t) = param
- `:tan` - tangent schedule: g(t) = (π/2) * sin((1-t)*π/2) / (cos((1-t)*π/2) + eps)
- `:linear` - linear: g(t) = param * t
- `Symbol("1-t/t")` - ratio: g(t) = (1-t) / (t + eps)
- `Symbol("1/t")` - inverse: g(t) = 1 / (t + eps)

Note: sde_gt_param is used for :const and :linear modes only.
For other modes, it's typically 1.0 (no scaling).

Returns clamped to [0, clamp_val] to prevent numerical issues.
"""
function _compute_gt(P::RDNFlow, t::T; clamp_val::T=T(1e5), eps::T=T(1e-2)) where T
    t_clamped = clamp(t, T(0), T(1) - T(1e-5))

    gt = if P.sde_gt_mode == :const
        T(P.sde_gt_param)
    elseif P.sde_gt_mode == :tan
        # Tangent schedule matching Python: (π/2) * sin((1-t)*π/2) / (cos((1-t)*π/2) + eps)
        num = sin((one(T) - t_clamped) * T(π/2))
        den = cos((one(T) - t_clamped) * T(π/2))
        T(π/2) * num / (den + eps)
    elseif P.sde_gt_mode == :linear
        T(P.sde_gt_param) * t_clamped
    elseif P.sde_gt_mode == Symbol("1-t/t")
        (one(T) - t_clamped) / (t_clamped + eps)
    elseif P.sde_gt_mode == Symbol("1/t")
        one(T) / (t_clamped + eps)
    else
        T(0)
    end

    return clamp(gt, T(0), clamp_val)
end

# Velocity to X1 prediction and vice versa
"""
    v_to_x1(x_t, v, t)

Convert velocity prediction to X1 prediction.
X1 = x_t + (1-t) * v
"""
function v_to_x1(x_t::AbstractArray{T}, v::AbstractArray{T}, t) where T
    t_exp = expand(t, ndims(x_t))
    return x_t .+ (one(T) .- t_exp) .* v
end

"""
    x1_to_v(x_t, x1, t)

Convert X1 prediction to velocity prediction.
v = (x1 - x_t) / (1 - t)
"""
function x1_to_v(x_t::AbstractArray{T}, x1::AbstractArray{T}, t; eps::T=T(1e-5)) where T
    t_exp = expand(t, ndims(x_t))
    return (x1 .- x_t) ./ (one(T) .- t_exp .+ eps)
end

# ============================================================================
# Step function for gen() integration
# ============================================================================

"""
    vf_to_score(x_t, v, t)

Convert velocity field to score function.
score(x_t, t) = (t * v - x_t) / (1 - t)
"""
function vf_to_score(x_t::AbstractArray{T}, v::AbstractArray{T}, t) where T
    t_exp = expand(t, ndims(x_t))
    eps = T(1e-5)
    return (t_exp .* v .- x_t) ./ (one(T) .- t_exp .+ eps)
end

"""
    step(P::RDNFlow, Xₜ::ContinuousState, X̂₁::ContinuousState, s₁, s₂)

Single step for flow matching. Moves from uniform progress s₁ to s₂.
X̂₁ is the predicted endpoint (X1 prediction from the model).

The actual interpolation times are τ(s₁) and τ(s₂) where τ is the schedule transform.
The step size in actual time is dτ = τ(s₂) - τ(s₁).

If P.sc_scale_noise > 0 and τ < P.t_lim_ode, uses SDE sampling:
    dx = [v + g(τ) * sc_scale_score * score] dτ + sqrt(2 * g(τ) * sc_scale_noise) dW

Otherwise uses deterministic ODE:
    dx = v * dτ

The noise schedule g(τ) is controlled by P.sde_gt_mode and P.sde_gt_param.
Switches to pure ODE when τ >= P.t_lim_ode.
"""
function step(P::RDNFlow, Xₜ::ContinuousState, X̂₁::ContinuousState, s₁, s₂)
    T = eltype(tensor(Xₜ))

    # s₁ and s₂ are uniform progress values; convert to actual interpolation times
    tau1 = schedule_transform(P, T(s₁))
    tau2 = schedule_transform(P, T(s₂))
    dtau = tau2 - tau1

    xt = tensor(Xₜ)
    x1 = tensor(X̂₁)

    # Velocity: v = (x1 - xt) / (1 - τ)
    # This is dx/dτ, so we integrate: x_new = xt + v * dτ
    v = x1_to_v(xt, x1, tau1)

    # Check if we should use SDE or ODE (based on actual time τ)
    use_sde = P.sc_scale_noise > T(1e-8) && tau1 < P.t_lim_ode

    if use_sde
        # Compute g(τ) for SDE noise
        gt = _compute_gt(P, tau1)

        # SDE step: dx = [v + g(τ) * sc_scale_score * score] dτ + sqrt(2 * g(τ) * sc_scale_noise) dW
        score = vf_to_score(xt, v, tau1)

        # Deterministic drift with scaled score
        drift = v .+ gt .* T(P.sc_scale_score) .* score

        # Stochastic noise with scaled diffusion
        noise = randn(T, size(xt))
        if P.zero_com
            noise = _force_zero_com(noise, nothing)
        end
        # Ensure diffusion coefficient is non-negative
        diffusion_var = max(T(0), T(2) * gt * T(P.sc_scale_noise) * abs(dtau))
        diffusion_coef = sqrt(diffusion_var)
        diffusion = diffusion_coef .* noise

        x_new = xt .+ drift .* dtau .+ diffusion
    else
        # Pure ODE step: dx = v * dτ
        x_new = xt .+ v .* dtau
    end

    # Enforce zero center of mass if needed
    if P.zero_com
        x_new = _force_zero_com(x_new, nothing)
    end

    return ContinuousState(x_new)
end

# Handle raw tensor predictions (resolveprediction converts these to ContinuousState)
function step(P::RDNFlow, Xₜ::ContinuousState, X̂₁::AbstractArray, s₁, s₂)
    step(P, Xₜ, ContinuousState(X̂₁), s₁, s₂)
end
