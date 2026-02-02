# RDN Flow: Flow matching on (R^d)^n
# For protein coordinates and latent spaces

using ForwardBackward

# Simple mean implementation (avoid Statistics dependency)
_mean(x; dims) = sum(x; dims=dims) ./ size(x, dims...)

"""
    RDNFlow(dim::Int; zero_com::Bool=false, sde_gt_mode::Symbol=:const, sde_gt_param::Real=0.0)

Flow matching process on (R^d)^n where n is the number of elements (e.g., residues)
and d is the dimensionality per element.

Uses linear interpolation for bridging: x_t = (1-t)*x_0 + t*x_1

# Arguments
- `dim`: Dimensionality d (e.g., 3 for CA coordinates, 8 for latents)
- `zero_com`: Whether to enforce zero center of mass (typically true for coordinates)
- `sde_gt_mode`: Mode for g(t) noise schedule in SDE inference (:const, :tan, :linear)
- `sde_gt_param`: Parameter for g(t) schedule

# Example
```julia
# For CA coordinates (3D, zero COM)
P_ca = RDNFlow(3; zero_com=true)

# For local latents (8D, no zero COM)
P_latent = RDNFlow(8; zero_com=false)

# Combined product space (use tuple)
P = (P_ca, P_latent)
```
"""
struct RDNFlow{T<:Real} <: Process
    dim::Int
    zero_com::Bool
    sde_gt_mode::Symbol
    sde_gt_param::T
end

RDNFlow(dim::Int; zero_com::Bool=false, sde_gt_mode::Symbol=:const, sde_gt_param::Real=0.0f0) =
    RDNFlow(dim, zero_com, sde_gt_mode, Float32(sde_gt_param))

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

# Bridge implementation - linear interpolation
# The default Deterministic bridge does this, but we override to handle zero-COM
function ForwardBackward.endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, P::RDNFlow, tF, tB)
    T = eltype(tF)
    d = ndims(X0.state)
    t0_exp = expand(tF ./ (tF .+ tB), d)  # Weight for X1
    t1_exp = expand(one(T) .- t0_exp, d)   # Weight for X0

    result = X0.state .* t1_exp .+ X1.state .* t0_exp

    if P.zero_com
        result = _force_zero_com(result, nothing)
    end

    return ContinuousState(result)
end

# 3-arg version for compatibility
function ForwardBackward.endpoint_conditioned_sample(X0::ContinuousState, X1::ContinuousState, P::RDNFlow, t)
    T = eltype(t)
    return endpoint_conditioned_sample(X0, X1, P, t, clamp.(one(T) .- t, T(0), T(1)))
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
    _compute_gt(P::RDNFlow, t)

Compute g(t) noise schedule value at time t.

Supported modes:
- `:const` - constant: g(t) = param
- `:tan` - tangent schedule: g(t) = param * tan(π/2 * (1-t))
- `:linear` - linear: g(t) = param * t
- `Symbol("1-t/t")` - ratio: g(t) = param * (1-t) / t
- `Symbol("1/t")` - inverse: g(t) = param / t
"""
function _compute_gt(P::RDNFlow, t::T) where T
    eps = T(1e-5)
    t_clamped = clamp(t, T(0), T(1) - eps)

    if P.sde_gt_mode == :const
        return T(P.sde_gt_param)
    elseif P.sde_gt_mode == :tan
        # Tangent schedule: high noise early, low noise late
        return T(P.sde_gt_param) * tan(T(π/2) * (one(T) - t_clamped))
    elseif P.sde_gt_mode == :linear
        return T(P.sde_gt_param) * t_clamped
    elseif P.sde_gt_mode == Symbol("1-t/t")
        return T(P.sde_gt_param) * (one(T) - t_clamped) / (t_clamped + eps)
    elseif P.sde_gt_mode == Symbol("1/t")
        return T(P.sde_gt_param) / (t_clamped + eps)
    else
        return T(0)
    end
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

Single step for flow matching. Moves from time s₁ to s₂.
X̂₁ is the predicted endpoint (X1 prediction from the model).

If P.sde_gt_param > 0, uses SDE sampling:
    dx = [v + g(t) * score] dt + sqrt(2 * g(t)) dW

Otherwise uses deterministic ODE:
    dx = v * dt

The noise schedule g(t) is controlled by P.sde_gt_mode and P.sde_gt_param.
"""
function step(P::RDNFlow, Xₜ::ContinuousState, X̂₁::ContinuousState, s₁, s₂)
    T = eltype(tensor(Xₜ))
    dt = T(s₂) - T(s₁)
    t = T(s₁)

    xt = tensor(Xₜ)
    x1 = tensor(X̂₁)

    # Velocity: v = (x1 - xt) / (1 - t)
    v = x1_to_v(xt, x1, t)

    # Compute g(t) for potential SDE noise
    gt = _compute_gt(P, t)

    if gt > T(1e-8)
        # SDE step: dx = [v + g(t) * score] dt + sqrt(2 * g(t)) dW
        score = vf_to_score(xt, v, t)

        # Deterministic drift
        drift = v .+ gt .* score

        # Stochastic noise
        noise = randn(T, size(xt))
        if P.zero_com
            noise = _force_zero_com(noise, nothing)
        end
        diffusion = sqrt(T(2) * gt * dt) .* noise

        x_new = xt .+ drift .* dt .+ diffusion
    else
        # Pure ODE step: dx = v * dt
        x_new = xt .+ v .* dt
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
