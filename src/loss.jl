mse(X̂₁, X₁) = abs2.(tensor(X̂₁) .- tensor(X₁)) #Mean Squared Error
lce(X̂₁, X₁) = -sum(tensor(X₁) .* logsoftmax(tensor(X̂₁)), dims=1) #Logit Cross Entropy
kl(P,Q) = sum(softmax(tensor(P)) .* (logsoftmax(tensor(P)) .- log.(tensor(Q))), dims=1) #Kullback-Leibler Divergence
rkl(P,Q) = sum(tensor(Q) .* (log.(tensor(Q)) .- logsoftmax(tensor(P))), dims=1) #Reverse Kullback-Leibler Divergence

function scaledmaskedmean(l::AbstractArray{T}, c::Union{AbstractArray, Real}, m::Union{AbstractArray, Real}) where T
    expanded_m = expand(m, ndims(l))
    T(mean(l .* expand(c, ndims(l)) .* expanded_m) / ((sum(expanded_m)/T(length(expanded_m))) + T(1e-6)))
end

scaledmaskedmean(l::AbstractArray, c::Union{AbstractArray, Real}, m::Nothing) = mean(l .* expand(c, ndims(l)))

#Need to consider loss scaling for discrete case, since we're predicting a distribution instead of a point.
#Similar to if we were allowed to predict a Gaussian variance, the scaling would semi-automatically compensate.
scalefloss(P::UProcess, t, pow = 2, eps = eltype(t)(0.05)) = 1 ./ ((1 + eps) .- tscale(P, t)) .^ pow
scalefloss(P::Tuple, t::Union{AbstractArray, Real}, pow = 2, eps = eltype(t)(0.05)) = scalefloss.(P, (t,), (pow,), (eps,))
scalefloss(P::Tuple, t::Tuple, pow = 2, eps = eltype(t)(0.05)) = scalefloss.(P, t, (pow,), (eps,))

"""
    floss(Xₜ::UState, X̂₁, X₁::UState, c)
    floss(P::UProcess, Xₜ::UState, X̂₁, X₁::UState, c)
    floss(P::UProcess, X̂₁, X₁::UState, c)
    
Where c is shaped like t, and scales the loss. Typical call with default loss scaling would be like: `floss(P, X̂₁, X₁, scalefloss(P, t))`
"""

fbu(T) = Union{T, FProcess{<:T}}
msu(T) = Union{T, MaskedState{<:T}}

floss(P::fbu(Deterministic),                X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(BrownianMotion),               X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::OUFlow,                            X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::OUBridgeExpVar,                    X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(VPCosineFlow),                 X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(ManifoldProcess{<:Euclidean}), X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
#floss(P::fbu(OrnsteinUhlenbeck),            X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁)) #<- I'm not sure MSE on X1 works for this process. We need to pull X1 back to Xt and get the generator.
#For a discrete process, X̂₁ will be a distribution, and X₁ will have to be a onehot before going onto the gpu.
floss(P::fbu(DiscreteProcess), X̂₁, X₁::msu(DiscreteState{<:AbstractArray{<:Integer}}), c) = error("X₁ needs to be onehot encoded with `onehot(X₁)`. You might need to do this before moving it to the GPU.")
floss(P::fbu(DiscreteProcess), X̂₁, X₁::msu(DiscreteState{<:OneHotArray}), c) = scaledmaskedmean(lce(X̂₁, X₁), c, getlmask(X₁))
floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Union{AbstractArray, Real}) = sum(floss.(P, X̂₁, X₁, (c,)))
floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Tuple) = sum(floss.(P, X̂₁, X₁, c))
floss(P::Union{fbu(ManifoldProcess), fbu(Deterministic)}, ξhat, ξ::Guide, c) = scaledmaskedmean(mse(ξhat, ξ.H), c, getlmask(ξ))


#I should make a self-balancing loss that tracks the running mean/std and adaptively scales to balance against target weights.

########################################################################
#Manifold-specific helper functions
########################################################################

########################################################################
#SO(3): Rotations
########################################################################

#It is sometimes useful to be able to compute the tangent coordinates from the predicted endpoint,
#eg. if the model needs to reason about its final location during the forward pass.
function so3_tangent_coordinates_stack(R1::AbstractArray{T,3}, Rt::AbstractArray{T,3}) where T
    eps = T(0.00001)
    R = batched_mul(batched_transpose(Rt), R1)
    tr_R = R[1,1,:] .+ R[2,2,:] .+ R[3,3,:]
    theta = acos.(clamp.((tr_R .- 1) ./ 2,T(-0.99),T(0.99)))
    sin_theta = sin.(theta)
    coeff = T(0.5) ./ (sin_theta .+ eps)
    axis_x = reshape(coeff .* (R[3,2,:] .- R[2,3,:]), 1, :)
    axis_y = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, :)
    axis_z = reshape(coeff .* (R[2,1,:] .- R[1,2,:]), 1, :)
    axis = vcat(axis_x, axis_y, axis_z)
    theta = reshape(theta, 1, :)
    tangent = sqrt(T(2)) .* (theta .* axis)
    return tangent
end

function so3_tangent_coordinates_stack(R1::AbstractArray{T,4}, Rt::AbstractArray{T,4}) where T
    return reshape(so3_tangent_coordinates_stack(reshape(R1, 3, 3, :), reshape(Rt, 3, 3, :)), 3, size(R1,3), size(R1,4))
end
