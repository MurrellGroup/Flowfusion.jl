struct FProcess{A,B}
    P::A #Process
    F::B #Time transform
end

UProcess = Union{Process,FProcess}

"""
    MaskedState(S::State, cmask, lmask)

Wraps a `State` with a conditioning mask (`cmask`) and a loss mask (`lmask`).

Conditioning mask behavior:

The typical use is that it makes sense, during training, to construct the conditioning mask on the training observation, `X1``.
During inference, the conditioning mask (and conditioned-upon state) has to be present on `X1`.
This dictates the behavior of the masking:
- When `bridge()` is called, the mask, and the state where `cmask=1`, are inherited from `X1`.
- When `gen()` is called, the state and mask will be propogated from `X0` through all of the `Xt`s.

Loss mask behavior:
- Where `lmask=0`, that observation (where the shape/size of the observation is determined by the difference in dimensions between the mask and the state) is not included in the loss.
"""
struct MaskedState{A,B,C}
    S::A     #State
    cmask::B #Conditioning mask. 1 = Xt=X1
    lmask::C #Loss mask.         1 = included in loss
end

Adapt.adapt_structure(to, MS::MaskedState{<:State}) = MaskedState(Adapt.adapt(to, MS.S), Adapt.adapt(to, MS.cmask), Adapt.adapt(to, MS.lmask))
Adapt.adapt_structure(to, MS::MaskedState{<:CategoricalLikelihood}) = MaskedState(Adapt.adapt(to, MS.S), Adapt.adapt(to, MS.cmask), Adapt.adapt(to, MS.lmask))

#For when we want to predict the transitions instead of X1hat
"""
    Guide(H::AbstractArray)

Wrapping a model prediction in Guide instructs the solver that the prediction points to X1 from the current state, instead of being a prediction of X1 itself.
Used for ManifoldStates where the prediction is a tangent 
"""
struct Guide{A, B, C}
    H::A
    cmask::B
    lmask::C
end

Guide(H) = Guide(H, nothing, nothing)

Adapt.adapt_structure(to, G::Guide) = Guide(Adapt.adapt(to, G.H), Adapt.adapt(to, G.cmask), Adapt.adapt(to, G.lmask))

UState = Union{State,MaskedState, Guide}

#This is for all Flow types where the mixture probabilities are directly defined, and the gen is done via probability velocities.
abstract type ConvexInterpolatingDiscreteFlow <: DiscreteProcess end #https://arxiv.org/pdf/2407.15595

# Edit-based discrete processes (insert/substitute/delete)
abstract type DiscreteIndelProcess <: DiscreteProcess end

struct InterpolatingDiscreteFlow <: ConvexInterpolatingDiscreteFlow
    κ::Function
    κ̇::Function
end

struct NoisyInterpolatingDiscreteFlow{T} <: ConvexInterpolatingDiscreteFlow
    κ₁::Function    # schedule for target token interpolation
    κ₂::Function    # schedule for uniform noise probability
    dκ₁::Function   # derivative of κ₁
    dκ₂::Function   # derivative of κ₂
    mask_token::T   # the token that is used for the X0 state
end

#A process where mean to which it reverts is X1
struct OUFlow{T} <: Process
    θ::T
    v_at_0::T
    v_at_1::T
    dec::T
end

OUFlow(θ::T, v_at_0::T) where T = OUFlow(θ, v_at_0, T(1e-2), T(-0.1))