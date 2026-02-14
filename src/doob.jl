#Note: Haven't figured out exactly what, in the literature, this is. Not very tested!

struct DoobMatchingFlow{Proc, B, F} <: Process
    P::Proc
    onescale::B #Controls whether the "step" is unit scale or "time remaining" scale. Need to think carefully about schedules in all this...
    transform::F #Transforms the output of the model to the rate space. Must act on the whole tensor.
    #Note: losses can be compared for different transforms, but not for different onescale.
end

DoobMatchingFlow(P::DiscreteProcess) = DoobMatchingFlow(P, true, NNlib.softplus) #x -> exp.(clamp.(x, -100, 11)) also works, but is scary
DoobMatchingFlow(P::DiscreteProcess, transform::Function) = DoobMatchingFlow(P, true, transform)
DoobMatchingFlow(P::DiscreteProcess, onescale::Bool) = DoobMatchingFlow(P, onescale, NNlib.softplus)

onescale(P::DoobMatchingFlow,t) = P.onescale ? (1 .- t)  : eltype(t)(1)
mulexpand(t,x) = expand(t, ndims(x)) .* x

#We could consider making this preserve one-hotness:
bridge(p::DoobMatchingFlow, x0::DiscreteState, x1::DiscreteState, t) = bridge(p.P, x0, x1, t)
bridge(p::DoobMatchingFlow, x0::DiscreteState, x1::DiscreteState, t0, t) = bridge(p.P, x0, x1, t0, t)

function fallback_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState; delta = eltype(t)(1e-5))
    return (tensor(forward(Xt, P, delta) ⊙ backward(X1, P, (1 .- t) .- delta)) .- tensor(onehot(Xt))) ./ delta;
end

doob_guide(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState) = fallback_doob(P, t, Xt, X1)

function closed_form_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState)
    tenXt = tensor(onehot(Xt))
    bk = tensor(backward(X1, P, 1 .- t))
    fv = forward_positive_velocities(onehot(Xt), P)
    positive_doob = (fv .* bk) ./ sum(bk .* tenXt, dims = 1)
    return positive_doob .- tenXt .* sum(positive_doob, dims = 1)
end

forward_positive_velocities(Xt::DiscreteState, P::PiQ)= (P.r .* (P.π ./ sum(P.π))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::PiQ, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)
forward_positive_velocities(Xt::DiscreteState, P::UniformUnmasking{T}) where T = (P.μ .* T((1 ./ (Xt.K-1)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformUnmasking, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)
forward_positive_velocities(Xt::DiscreteState, P::UniformDiscrete{T}) where T = (P.μ * T(1/(Xt.K*(1-1/Xt.K)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformDiscrete, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

Guide(P::DoobMatchingFlow, t, Xt::DiscreteState, X1::DiscreteState) = Flowfusion.Guide(mulexpand(onescale(P, t), doob_guide(P.P, t, Xt, X1)))
Guide(P::DoobMatchingFlow, t, mXt::Union{MaskedState{<:DiscreteState}, DiscreteState}, mX1::MaskedState{<:DiscreteState}) = Guide(mulexpand(onescale(P, t), doob_guide(P.P, t, mXt, mX1)), mX1.cmask, mX1.lmask)

function rate_constraint(Xt, X̂₁, f) 
    posQt = f(X̂₁) .* (1 .- Xt)   
    diagQt = -sum(posQt, dims = 1) .* Xt
    return posQt .+ diagQt
end

function velo_step(P, Xₜ::DiscreteState, delta_t, log_velocity, scale)
    ohXₜ = onehot(Xₜ)
    velocity = rate_constraint(tensor(ohXₜ), log_velocity, P.transform) .* scale
    newXₜ = CategoricalLikelihood(eltype(delta_t).(tensor(ohXₜ) .+ (delta_t .* velocity)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end

step(P::DoobMatchingFlow, Xₜ::DiscreteState, veloX̂₁::Flowfusion.Guide, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁.H, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁.H)))
step(P::DoobMatchingFlow, Xₜ::DiscreteState, veloX̂₁, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁)))

function cgm_dloss(P, Xt, X̂₁, doobX₁)
    Qt = P.transform(X̂₁)
    return sum((1 .- Xt) .* (Qt .- xlogy.(doobX₁, Qt)), dims = 1) #<- note, diagonals ignored; implicit zero sum
end

floss(P::Flowfusion.fbu(DoobMatchingFlow), Xt::Flowfusion.msu(DiscreteState), X̂₁, X₁::Guide, c) = Flowfusion.scaledmaskedmean(cgm_dloss(P, tensor(Xt), tensor(X̂₁), X₁.H), c, Flowfusion.getlmask(X₁))
