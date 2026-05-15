##########################################
#For processes that aren't used elsewhere
##########################################

function _vp_check_flow_time(t::Real)
    0 <= t <= 1 || throw(ArgumentError("flow time must be in [0, 1], got $t"))
    return nothing
end

function _vp_check_flow_time(t::AbstractArray)
    all((0 .<= t) .& (t .<= 1)) ||
        throw(ArgumentError("all flow times must be in [0, 1]"))
    return nothing
end

function _vp_check_clean_endpoint(t)
    vals = t isa Real ? (t,) : t
    all(x -> isapprox(float(x), 1.0; atol=sqrt(eps(float(x)))), vals) ||
        throw(ArgumentError("VPCosineFlow expects endpoint time 1"))
    return nothing
end

"""
    vp_alpha_bar(P::VPCosineFlow, t)

Return the cumulative signal power of the cosine VP schedule at flow time `t`.
Flow time is oriented so that `t=0` is noisiest and `t=1` is the clean endpoint.
"""
function vp_alpha_bar(P::VPCosineFlow, t)
    _vp_check_flow_time(t)
    diffusion_index = (1 .- t) .* P.n_timestep
    angle = diffusion_index ./ (P.n_timestep + 1) .* (pi / 2)
    return cos.(angle) .^ 2
end

"""
    vp_bridge_coefficients(P::VPCosineFlow, s, t)

Coefficients for the exact endpoint-conditioned transition `x_t | x_s, x_1`
under the VP schedule, for flow times `0 <= s <= t <= 1`.

Returns `(coef_x1, coef_xs, variance)` such that
`x_t = coef_x1 * x_1 + coef_xs * x_s + sqrt(variance) * z`.
The inputs may be scalars or broadcast-compatible arrays.
"""
function vp_bridge_coefficients(P::VPCosineFlow, s, t)
    _vp_check_flow_time(s)
    _vp_check_flow_time(t)
    all(s .<= t) || throw(ArgumentError("expected s <= t for x_t | x_s, x_1"))

    A_s = vp_alpha_bar(P, s)
    A_t = vp_alpha_bar(P, t)
    denom = max.(1 .- A_s, eps(Float64))
    ratio = clamp.(A_s ./ A_t, 0, 1)

    coef_x1 = sqrt.(A_t) .* (1 .- ratio) ./ denom
    coef_xs = sqrt.(ratio) .* (1 .- A_t) ./ denom
    variance = (1 .- A_t) .* (1 .- ratio) ./ denom
    return coef_x1, coef_xs, max.(variance, 0)
end

function ForwardBackward.endpoint_conditioned_sample(
    Xa::ContinuousState,
    Xc::ContinuousState,
    P::VPCosineFlow,
    t_a,
    t_b,
    t_c,
)::ContinuousState
    size(Xa.state) == size(Xc.state) ||
        throw(DimensionMismatch("Xa and Xc must have the same state shape"))
    _vp_check_clean_endpoint(t_c)

    xa = Xa.state
    xc = Xc.state
    nd = ndims(xa)
    ta = expand(t_a, nd)
    tb = expand(t_b, nd)
    _vp_check_flow_time(ta)
    _vp_check_flow_time(tb)
    all(ta .<= tb) || throw(ArgumentError("expected t_a <= t_b"))

    coef_x1, coef_xa, variance = vp_bridge_coefficients(P, ta, tb)
    mu = coef_x1 .* xc .+ coef_xa .* xa
    noise = randn(eltype(xa), size(xa)...)
    xb = mu .+ sqrt.(variance) .* noise
    return ContinuousState(xb)
end

##########################################
#https://arxiv.org/pdf/2407.15595
##########################################

#Note to future self: if I ever use the probability velocity trick for one of the regular discrete CTMCs (thus turning it into a ConvexInterpolatingDiscreteFlow), I'll need to handle the FProcess schedule a little differently.
FProcess(p::ConvexInterpolatingDiscreteFlow, f) = Error("ConvexInterpolatingDiscreteFlow have their own schedule mechanisms. Do not use them with FProcess.")

"""
    InterpolatingDiscreteFlow(κ::Function, κ̇::Function)
    InterpolatingDiscreteFlow() - Uses default Cosine scheduler.


A Discrete process that interpolates between two states (equation 9 from https://arxiv.org/pdf/2407.15595)
κ controls the interpolation schedule, κ̇ is the derivative of κ.
Works when model predicts `X̂₁` with cross-entropy loss (`floss` will do this).
"""

InterpolatingDiscreteFlow() = InterpolatingDiscreteFlow(t -> 1-cos((pi/2)*t), t -> (pi/2)*sin((pi/2)*t))

#The weird type sig here is to avoid dispatch on onehot arrays.
function bridge(p::InterpolatingDiscreteFlow, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t)
    ts = expand(t, ndims(x0.state))
    i = p.κ.(ts) .≥ rand(size(x0.state)...)
    xt = copy(x0)
    xt.state[i] .= x1.state[i]
    return xt
end

function step(P::InterpolatingDiscreteFlow, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, X̂₁, s₁, s₂)
    step = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    velo = (P.κ̇.(s₁) ./ (1 - P.κ.(s₁))) .* (tensor(X̂₁) - tensor(ohXₜ))
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (step .* velo)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end


"""
    NoisyInterpolatingDiscreteFlow(κ₁, κ₂, dκ₁, dκ₂, dummy_token)
    NoisyInterpolatingDiscreteFlow(noise; K = 1, dummy_token = nothing) - Uses default cosine schedule, where `noise` is the maximum amplitude of the uniform noise component.
    NoisyInterpolatingDiscreteFlow() - Uses default cosine schedule and noise = 0.2.

A convex mixture of X0, uniform noise, and X1. Equation 10 in https://arxiv.org/pdf/2407.15595
Compared to InterpolatingDiscreteFlow, it encourages the model to make multiple switches during inference.
κ₁, κ₂ are the schedules for target token interpolation and uniform noise probability.
dκ₁, dκ₂ are the derivatives of κ₁, κ₂.
Defaults to using a cosine schedule. `K=2` will resolve the discrete states later than `K=1`.
If K>1 things might break if your X0 is not the `dummy_token` (also called the masked token) which should be passed to NoisyInterpolatingDiscreteFlow.
"""
function NoisyInterpolatingDiscreteFlow(noise; K = 1, dummy_token::T = nothing) where T
    if (K > 1 && isnothing(dummy_token)) 
        @warn "NoisyInterpolatingDiscreteFlow: If K>1 things might break if your X0 is not the `dummy_token` (which should also be passed to NoisyInterpolatingDiscreteFlow)."
    end
    return NoisyInterpolatingDiscreteFlow{T}(
                t -> oftype(t,(1 - cos((π/2)*t))^K), #K1
                t -> oftype(t,(noise * sin(π*t))), #K2
                t -> oftype(t,(K * (π/2) * sin((π/2) * t) * (1 - cos((π/2) * t))^(K - 1))), #dK1
                t -> oftype(t,(noise*π*cos(π*t))), #dK2
                dummy_token
                )
end
NoisyInterpolatingDiscreteFlow() = NoisyInterpolatingDiscreteFlow{Nothing}(0.2)
NoisyInterpolatingDiscreteFlow(noise, power) = NoisyInterpolatingDiscreteFlow(noise, K = power)
function bridge(p::NoisyInterpolatingDiscreteFlow, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t)
    D = size(x0.state)
    ts = expand(t, ndims(x0.state))
    Xt = copy(x0)
    rands = rand(D...) 
    x1bool = p.κ₁.(ts) .> rands
    uniformbool = (p.κ₂.(ts) .+ p.κ₁.(ts)) .> rands
    for idx in eachindex(rands)
        if x1bool[idx]
            Xt.state[idx] = x1.state[idx]
        elseif uniformbool[idx]
            Xt.state[idx] = rand(1:x0.K)
        else
            Xt.state[idx] = x0.state[idx]
        end
    end
    return Xt
end
function step(P::NoisyInterpolatingDiscreteFlow{Nothing}, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, X̂₁, s₁, s₂)
    T = eltype(s₁)
    Δt = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    pu = T(1/Xₜ.K)
    eps = T(1e-10)
    κ1 = P.κ₁.(s₁)
    κ2 = P.κ₂.(s₁)
    κ3 = (1 .- (κ1 .+ κ2))  # κ₃(t)=1-κ₁(t)-κ₂(t)
    dκ1 = P.dκ₁.(s₁)
    dκ2 = P.dκ₂.(s₁)
    dκ3 = .- (dκ1 .+ dκ2)  # Because dκ₃ = - (dκ₁+dκ₂)
    bt = dκ3 ./ (eps .+ κ3)
    #Theorem 3 applied to equation 10 in https://arxiv.org/pdf/2407.15595
    velo = (dκ1 .- κ1 .* bt) .* tensor(X̂₁) .+ (dκ2 .- κ2 .* bt) .* pu .+ bt .* tensor(ohXₜ)
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (Δt .* velo)))
    clamp!(tensor(newXₜ), 0, Inf)
    return rand(newXₜ)
end
function step(P::NoisyInterpolatingDiscreteFlow{<:Integer}, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, X̂₁, s₁, s₂)
    T = eltype(s₁)
    Δt = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    pu = T(1/Xₜ.K)
    eps = T(1e-10)
    κ1 = P.κ₁.(s₁)
    κ2 = P.κ₂.(s₁)
    κ3 = (1 .- (κ1 .+ κ2))  # κ₃(t)=1-κ₁(t)-κ₂(t)
    dκ1 = P.dκ₁.(s₁)
    dκ2 = P.dκ₂.(s₁)
    dκ3 = .- (dκ1 .+ dκ2)  # Because dκ₃ = - (dκ₁+dκ₂)
    #Theorem 3 applied to equation 10 in https://arxiv.org/pdf/2407.15595
    r1 = dκ1 ./ (eps .+ κ1)
    r2 = dκ2 ./ (eps .+ κ2)
    r3 = dκ3 ./ (eps .+ κ3)
    bt = min.(r1,r2, r3) #b_t = min_j dκ_j/κ_j
    a1 = dκ1 .- κ1 .* bt             # component 1 (denoiser)
    a2 = dκ2 .- κ2 .* bt             # component 2 (uniform)
    a3 = dκ3 .- κ3 .* bt             # component 3 (dummy/mask)
    velo =  a1 .* tensor(X̂₁) .+ a2 .* pu .+ bt .* tensor(ohXₜ)
    selectdim(velo,1,P.mask_token) .+= a3 #Adding the mask token compoenent to the correct tensor slice
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (Δt .* velo)))
    clamp!(tensor(newXₜ), 0, Inf)
    return rand(newXₜ)
end

function bridge(P::OUFlow, X0, X1, t0, t)
    OU = OrnsteinUhlenbeckExpVar(tensor(X1), P.θ, P.v_at_0, P.v_at_1, dec = P.dec) #<-Note X1 as mean
    endpoint_conditioned_sample(X0, X1, OU, t0, t, eltype(t)(1))
end
