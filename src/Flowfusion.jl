#=
Need to test/do:
Urgent:
- Test tuples!
- Test Manifolds with masking (especially tangent_guide and apply_tangent etc) 
- Masking (cmask) on all state types for bridge and gen
- Masking (lmask) on all state types for both losses
- tensor on masked states
- FProcess and whether it matches the target where allowed. Need to come up with a policy on using FProcess with InterpolatingDiscreteProcesses
- X1 pred for rotations (add angle/axis loss back in just because yolo)
- self-conditioning
- GPU use of all state types
Later:
- Make a table of Manifolds where you test whether the key functions are defined, with checkboxes and timing for diffusion and flow.
- Make a table of commands for key types of diffusion/flow. Columns for Process, X0/X1 setup, Xt bridge, loss, gen where things like softmax, Guide, etc are clear.
- Compute probability velocities for UniformDiscrete and PiQ so these can flow.
=#


module Flowfusion

using ForwardBackward, OneHotArrays, Adapt, Manifolds, NNlib, LogExpFunctions, Distributions

include("types.jl")
include("mask.jl")
include("bridge.jl")
include("loss.jl")
include("processes.jl")
include("doob.jl")

include("batching.jl")

include("indel.jl")
include("dist_dfm.jl")
include("rdn_flow.jl")

export
    #Processes not in ForwardBackward.jl
    InterpolatingDiscreteFlow,
    NoisyInterpolatingDiscreteFlow,
    DoobMatchingFlow,
    OUFlow,
    RDNFlow,
    sample_rdn_noise,
    schedule_transform,
    rdn_bridge,
    rdn_sde_step,
    v_to_x1,
    x1_to_v,
    vf_to_score,
    MaskedState,
    Guide,
    tangent_guide,
    bridge,
    step,
    scalefloss,
    gen,
    Tracker,
    stack_tracker,
    onehot,
    unhot,
    FProcess,
    floss,
    tcloss,
    dense,
    batch,
    regroup,
    combine


#Useful for demos etc:
#Define a cat - from https://www.geogebra.org/m/pH8wD3rW
cat_shape(t) = [-(721*sin(t))/4+196/3*sin(2*t)-86/3*sin(3*t)-131/2*sin(4*t)+477/14*sin(5*t)+27*sin(6*t)-29/2*sin(7*t)+68/5*sin(8*t)+1/10*sin(9*t)+23/4*sin(10*t)-19/2*sin(12*t)-85/21*sin(13*t)+2/3*sin(14*t)+27/5*sin(15*t)+7/4*sin(16*t)+17/9*sin(17*t)-4*sin(18*t)-1/2*sin(19*t)+1/6*sin(20*t)+6/7*sin(21*t)-1/8*sin(22*t)+1/3*sin(23*t)+3/2*sin(24*t)+13/5*sin(25*t)+sin(26*t)-2*sin(27*t)+3/5*sin(28*t)-1/5*sin(29*t)+1/5*sin(30*t)+(2337*cos(t))/8-43/5*cos(2*t)+322/5*cos(3*t)-117/5*cos(4*t)-26/5*cos(5*t)-23/3*cos(6*t)+143/4*cos(7*t)-11/4*cos(8*t)-31/3*cos(9*t)-13/4*cos(10*t)-9/2*cos(11*t)+41/20*cos(12*t)+8*cos(13*t)+2/3*cos(14*t)+6*cos(15*t)+17/4*cos(16*t)-3/2*cos(17*t)-29/10*cos(18*t)+11/6*cos(19*t)+12/5*cos(20*t)+3/2*cos(21*t)+11/12*cos(22*t)-4/5*cos(23*t)+cos(24*t)+17/8*cos(25*t)-7/2*cos(26*t)-5/6*cos(27*t)-11/10*cos(28*t)+1/2*cos(29*t)-1/5*cos(30*t),
    -(637*sin(t))/2-188/5*sin(2*t)-11/7*sin(3*t)-12/5*sin(4*t)+11/3*sin(5*t)-37/4*sin(6*t)+8/3*sin(7*t)+65/6*sin(8*t)-32/5*sin(9*t)-41/4*sin(10*t)-38/3*sin(11*t)-47/8*sin(12*t)+5/4*sin(13*t)-41/7*sin(14*t)-7/3*sin(15*t)-13/7*sin(16*t)+17/4*sin(17*t)-9/4*sin(18*t)+8/9*sin(19*t)+3/5*sin(20*t)-2/5*sin(21*t)+4/3*sin(22*t)+1/3*sin(23*t)+3/5*sin(24*t)-3/5*sin(25*t)+6/5*sin(26*t)-1/5*sin(27*t)+10/9*sin(28*t)+1/3*sin(29*t)-3/4*sin(30*t)-(125*cos(t))/2-521/9*cos(2*t)-359/3*cos(3*t)+47/3*cos(4*t)-33/2*cos(5*t)-5/4*cos(6*t)+31/8*cos(7*t)+9/10*cos(8*t)-119/4*cos(9*t)-17/2*cos(10*t)+22/3*cos(11*t)+15/4*cos(12*t)-5/2*cos(13*t)+19/6*cos(14*t)+7/4*cos(15*t)+31/4*cos(16*t)-cos(17*t)+11/10*cos(18*t)-2/3*cos(19*t)+13/3*cos(20*t)-5/4*cos(21*t)+2/3*cos(22*t)+1/4*cos(23*t)+5/6*cos(24*t)+3/4*cos(26*t)-1/2*cos(27*t)-1/10*cos(28*t)-1/3*cos(29*t)-1/19*cos(30*t)]

random_literal_cat(dims...; sigma = 0.05f0) = typeof(sigma).(stack([cat_shape(rand()*2pi)/200 for _ in zeros(dims...)]) .+ randn(2, dims...) * sigma)

function discretize(x, d, lo, hi)
    for (i, v) in enumerate(range(lo, hi, length = d - 1))
        x < v && return i
    end
    d
end

random_discrete_cat(dims...; d = 32, lo = -2.5, hi = 2.5) = discretize.(random_literal_cat(dims...), (d,), (lo,), (hi,))

end
