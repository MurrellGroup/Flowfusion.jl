using Flowfusion
using Test
using Manifolds
using ForwardBackward

@testset "Flowfusion.jl" begin

    @testset "Masking" begin
        siz = (5,6,7)
        XC() = ContinuousState(randn(5, siz...))
        XD() = DiscreteState(5, rand(1:5, siz...))
        MT = Torus(2) #Where the representation is a vector
        XT() = ManifoldState(MT, [rand(MT) for _ in zeros(siz...)])
        MR = SpecialOrthogonal(3) #Where the representation is a matrix
        XR() = ManifoldState(MR, [rand(MR) for _ in zeros(siz...)])
        XDL() = CategoricalLikelihood(rand(5, siz...))
        XGL() = GaussianLikelihood(randn(5, siz...), randn(5, siz...), zeros(siz...))

        for f in [XC, XD, XT, XR, XDL, XGL] 
            Xa = f()
            Xb = f()
            Xc = Flowfusion.mask(Xa, Xb)
            @test isapprox(tensor(Xa),tensor(Xc))
            @test typeof(Xc) == typeof(Xa)
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xb, m, m)
            Xc = Flowfusion.mask(Xa, XM)

            @test typeof(Xc) == typeof(XM) #If you mask a regular State with a MaskedState, the result is a MaskedState.
            d = (tensor(Xb) .- tensor(Xc))
            @test isapprox(sum(d .* expand(.!m, ndims(d))),0)

            m = rand(Bool, siz...)
            Xa = MaskedState(f(), m, m)
            Xb = MaskedState(f(), m, m)
            Xc = Flowfusion.mask(Xa, Xb)
            @test typeof(Xc) == typeof(Xa)
            d = (tensor(Xb) .- tensor(Xc))
            @test isapprox(sum(d .* expand(.!m, ndims(d))),0)
        end
    end

    include("bridge_step_equivalence.jl")

    @testset "Bridge, step" begin

        siz = (5,6)
        XC() = ContinuousState(randn(5, siz...))
        XD() = DiscreteState(5, rand(1:5, siz...))
        MT = Torus(2)
        XT() = ManifoldState(MT, [rand(MT) for _ in zeros(siz...)])
        MR = SpecialOrthogonal(3)
        XR() = ManifoldState(MR, [rand(MR) for _ in zeros(siz...)])

        for (f,p) in [(XC, BrownianMotion()),
                    (XT, ManifoldProcess(1)),
                    (XR, ManifoldProcess(1)),
                    (XD, InterpolatingDiscreteFlow())]
            #bridge - propogates the mask
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xb, m, m)
            Xt = Flowfusion.bridge(p, Xa, XM, 0.1)
            @test typeof(Xt) == typeof(XM)
            if !(p isa InterpolatingDiscreteFlow)
                @test isapprox(sum((tensor(Xt) .== tensor(Xb))), sum(.!m) * (length(tensor(Xb)) / length(m)))
            else
                @test sum((tensor(Xt) .== tensor(Xb))) >= sum(.!m) * (length(tensor(Xb)) / length(.!m))
            end

            #step - doesn't propogate the mask
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xa, m, m)
            if !(p isa InterpolatingDiscreteFlow)
                Xt = Flowfusion.step(p, XM, Xa, 0.1, 0.1)
                @test isapprox(sum(tensor(Xt) .!= tensor(XM)), 0) #Because step size is zero
            else
                Xt = Flowfusion.step(p, XM, onehot(Xa), 0.1, 0.1)
                @test isapprox(sum(tensor(Xt) .!= tensor(XM)), 0) #Because step size is zero
            end
        end

    end
end
