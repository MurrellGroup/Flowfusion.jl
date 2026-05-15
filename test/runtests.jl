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

    @testset "Bridge, step" begin

        siz = (5,6)
        XC() = ContinuousState(randn(5, siz...))
        XD() = DiscreteState(5, rand(1:5, siz...))
        MT = Torus(2)
        XT() = ManifoldState(MT, [rand(MT) for _ in zeros(siz...)])
        MR = SpecialOrthogonal(3)
        XR() = ManifoldState(MR, [rand(MR) for _ in zeros(siz...)])

        for (f,p) in [(XC, BrownianMotion()),
                    (XC, VPFlow()),
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

    @testset "VP flow" begin
        P = VPFlow(CosineVPSchedule(1000))

        @test isapprox(vp_alpha_bar(P, 1.0), 1.0; atol=1e-12)
        @test vp_alpha_bar(P, 0.0) < 3e-6
        @test 0.49 < vp_alpha_bar(P, 0.5) < 0.51
        @test_throws ArgumentError vp_alpha_bar(P, -0.01)
        @test_throws ArgumentError vp_alpha_bar(P, 1.01)
        @test vp_alpha_bar(VPFlow(t -> t), 0.25) == 0.25

        for (s, t, u) in ((0.0, 0.2, 0.7), (0.05, 0.4, 0.95), (0.33, 0.66, 1.0))
            a_st, b_st, v_st = vp_bridge_coefficients(P, s, t)
            a_tu, b_tu, v_tu = vp_bridge_coefficients(P, t, u)
            a_su, b_su, v_su = vp_bridge_coefficients(P, s, u)

            @test isapprox(b_tu * b_st, b_su; rtol=1e-10, atol=1e-10)
            @test isapprox(a_tu + b_tu * a_st, a_su; rtol=1e-10, atol=1e-10)
            @test isapprox((b_tu^2) * v_st + v_tu, v_su; rtol=1e-10, atol=1e-10)
        end

        Xa = ContinuousState(randn(Float32, 2, 3, 4))
        X1 = ContinuousState(randn(Float32, 2, 3, 4))

        Xt = bridge(P, Xa, X1, 0.2f0)
        @test size(tensor(Xt)) == size(tensor(Xa))

        Xsame = bridge(P, Xt, X1, 0.4f0, 0.4f0)
        @test tensor(Xsame) ≈ tensor(Xt)

        t0 = Float32[0.0, 0.1, 0.2, 0.3]
        t1 = Float32[0.5, 0.6, 0.7, 0.8]
        Xvec = bridge(P, Xa, X1, t0, t1)
        @test size(tensor(Xvec)) == size(tensor(Xa))

        Xclean = bridge(P, Xa, X1, 0.3, 1.0)
        @test tensor(Xclean) ≈ tensor(X1)

        @test_throws ArgumentError bridge(P, Xa, X1, 0.5, 0.4)
        @test_throws ArgumentError ForwardBackward.endpoint_conditioned_sample(Xa, X1, P, 0.0, 0.5, 0.9)

        P_linear = VPFlow(t -> t)
        Xt0 = bridge(P_linear, Xa, X1, 0.0, 0.0)
        @test tensor(Xt0) ≈ tensor(Xa)
        Xlin = bridge(P_linear, Xa, X1, 0.0, 0.6)
        @test size(tensor(Xlin)) == size(tensor(Xa))
        Xlin_clean = bridge(P_linear, Xa, X1, 0.25, 1.0)
        @test tensor(Xlin_clean) ≈ tensor(X1)

        P_bad = VPFlow(t -> 1 - t)
        @test_throws ArgumentError bridge(P_bad, Xa, X1, 0.2, 0.4)
    end
end
