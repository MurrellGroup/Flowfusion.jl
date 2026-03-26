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

    @testset "Batching padded generic" begin
        # Build variable-length samples for each state type and test batch for plain and masked inputs
        sizA = (3,)   # feature dims for continuous/discrete
        MT = Torus(2)
        MR = SpecialOrthogonal(3)

        # helpers to make variable-length sequences
        function mkC(len)
            ContinuousState(randn(3, len))
        end
        function mkD(len)
            DiscreteState(5, rand(1:5, len))
        end
        function mkT(len)
            ManifoldState(MT, [rand(MT) for _ in 1:len])
        end
        function mkR(len)
            ManifoldState(MR, [rand(MR) for _ in 1:len])
        end

        lens = [2,4,3]
        # plain states
        for maker in (mkC, mkD, mkT, mkR)
            xs = [maker(l) for l in lens]
            B = Flowfusion.batch(xs)
            @test B isa MaskedState
            # masks: last dim is length, batch dim after that
            @test size(B.lmask) == (maximum(lens), length(lens))
            for i in eachindex(lens)
                @test all(B.lmask[1:lens[i], i])
                if lens[i] < maximum(lens)
                    @test all(.!B.lmask[(lens[i]+1):end, i])
                end
            end
        end

        # masked inputs: create per-item masks of their true length
        for maker in (mkC, mkD, mkT, mkR)
            xs = [maker(l) for l in lens]
            cms = [trues(l) for l in lens]
            lms = [falses(l) for l in lens]
            # flip some
            cms[2][end] = false
            lms[1][1] = true
            mxs = [MaskedState(xs[i], cms[i], lms[i]) for i in eachindex(xs)]
            B = Flowfusion.batch(mxs)
            @test B isa MaskedState
            @test size(B.cmask) == (maximum(lens), length(lens))
            @test B.cmask[lens[2], 2] == false
            @test B.lmask[1, 1] == true
            # padded zones must be false
            for i in eachindex(lens)
                if lens[i] < maximum(lens)
                    @test all(B.cmask[(lens[i]+1):end, i] .== false)
                    @test all(B.lmask[(lens[i]+1):end, i] .== false)
                end
            end
        end

        # tuple of states
        xs_tuple = [(mkC(l), mkD(l)) for l in lens]
        BT = Flowfusion.batch(xs_tuple)
        @test BT isa Tuple
        @test all(BT[i] isa MaskedState for i in 1:2)
        @test size(BT[1].lmask) == (maximum(lens), length(lens))
        @test size(BT[2].lmask) == (maximum(lens), length(lens))
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

    @testset "PoissonIndelProcess" begin
        @testset "branch_prob_ud!: trivial cases with α=0 (identity substitutions)" begin
            rng = Flowfusion.MersenneTwister(0)
            K = 5
            λ = 0.7
            μ = 1.3
            α = 0.0             # Pdiag=1, Poff=0
        
            p = UniformDiscretePoissonIndelProcess(λ, μ, α, K)
            t = 0.25
            C = Flowfusion.make_precomp(p, t)
            B = Flowfusion.make_branch_ud(p, C)
        
            # helpers
            s2 = C.s2
            Iins = B.Iins
        
            # F matrices sized per problem
            F = zeros(Float64, 1, 1)
            @test Flowfusion.branch_prob_ud!(F, Int[], Int[], B) ≈ 1.0 atol=1e-14
        
            F = zeros(Float64, 2, 1) # (n+1)×(m+1) for n=1,m=0
            @test Flowfusion.branch_prob_ud!(F, [1], Int[], B) ≈ (1 - s2) atol=1e-12
        
            F = zeros(Float64, 1, 2) # n=0,m=1
            @test Flowfusion.branch_prob_ud!(F, Int[], [2], B) ≈ Iins atol=1e-12
        
            # root=[1], leaf=[2] (≠): only del+ins; two orders: delete-then-insert, insert-then-delete
            F = zeros(Float64, 2, 2)
            @test Flowfusion.branch_prob_ud!(F, [1], [2], B) ≈ 2*(1 - s2)*Iins atol=1e-12
        
            # root=[3], leaf=[3] (==): direct match + two del/ins orders
            F = zeros(Float64, 2, 2)
            @test Flowfusion.branch_prob_ud!(F, [3], [3], B) ≈ (s2 + 2*(1 - s2)*Iins) atol=1e-12
        end

  @testset "PIP step (Euler, no batch)" begin
    rng = Flowfusion.MersenneTwister(123)
    Flowfusion.Random.seed!(rng, 0)
    K = 5
    p = UniformDiscretePoissonIndelProcess(0.1, 0.2, 0.3, K)
    # simple sequence
    Xt = DiscreteState(K, [1, 2, 3])
    n = length(tensor(Xt))
    dt = 1e-4

    # Case 1: pure deletion on position 2 at rate r, others zero
    del = zeros(Float64, n); del[2] = 7.0
    sub = zeros(Float64, K, n)
    ins = zeros(Float64, K, n + 1)
    guide = Flowfusion.Guide((sub = sub, del = del, ins = ins))
    # Repeat many trials to estimate probability of deletion ≈ dt * r
    trials = 100000
    cnt_delete = 0
    for _ in 1:trials
      Y = Flowfusion.step(p, Xt, guide, 0.0, dt)
      if length(tensor(Y)) == 2 && tensor(Y) == [1, 3]
        cnt_delete += 1
      end
    end
    #@test isapprox(cnt_delete / trials, dt * del[2]; rtol = 0.25)
    @show log10(cnt_delete / trials) , log10(dt * del[2])
    @test abs(log10(cnt_delete / trials) - log10(dt * del[2])) ≤ 0.1

    # Case 2: pure substitution at position 1 from token 1 to token 4 at rate r
    del .= 0
    sub .= 0
    sub[4, 1] = 11.0
    guide = Flowfusion.Guide((sub = sub, del = del, ins = ins))
    cnt_sub = 0
    for _ in 1:trials
      Y = Flowfusion.step(p, Xt, guide, 0.0, dt)
      if length(tensor(Y)) == 3 && tensor(Y)[1] == 4
        cnt_sub += 1
      end
    end
    @show log10(cnt_sub / trials) , log10(dt * sub[4, 1])
    @test abs(log10(cnt_sub / trials) - log10(dt * sub[4, 1])) ≤ 0.1

    # Case 3: pure insertion between positions 1 and 2 (gap s=1) of token 5 at rate r
    del .= 0
    sub .= 0
    ins .= 0
    ins[5, 2] = 13.0 # s=1 -> index 2 in (K, n+1)
    guide = Flowfusion.Guide((sub = sub, del = del, ins = ins))
    cnt_ins = 0
    for _ in 1:trials
      Y = Flowfusion.step(p, Xt, guide, 0.0, dt)
      y = tensor(Y)
      if length(y) == 4 && y[2] == 5
        cnt_ins += 1
      end
    end
    @show log10(cnt_ins / trials) , log10(dt * ins[5, 2])
    @test abs(log10(cnt_ins / trials) - log10(dt * ins[5, 2])) ≤ 0.1

    # Case 4: ensure at most one event per small dt when multiple hazards present
    del .= 2.0
    sub .= 0
    ins .= 1.0
    guide = Flowfusion.Guide((sub = sub, del = del, ins = ins))
    # For small dt, probability of >=2 events is o(dt); check empirically remains very small
    overtwo = 0
    trials2 = 20000
    for _ in 1:trials2
      Y = Flowfusion.step(p, Xt, guide, 0.0, dt)
      diff = length(tensor(Y)) - length(tensor(Xt))
      # deletions reduce len by >=1; insertions increase by >=1; substitutions don't change length
      if abs(diff) >= 2
        overtwo += 1
      end
    end
    @test overtwo / trials2 ≤ 5e-3
  end
        
        @testset "sample_alignment_ud: deterministic kernel" begin
            rng = Flowfusion.MersenneTwister(1)
            # Force only R moves when tokens match; no A/B, no off-diagonal R
            KUD = Flowfusion.KernelsUD{Float64}(0.0, 0.0, 1.0, 0.0)
        
            A = [1,2,3]
            B = [1,2,3]
            events, h = Flowfusion.sample_alignment_ud(rng, A, B, KUD)
            @test h ≈ 1.0 atol=1e-14
            @test all(ev.typ === :R for ev in events)
            @test length(events) == 3
            @test all(A[events[i].i] == B[events[i].j] == i for i in 1:3)
        
            # If sequences differ, forward mass should be zero
            A2 = [1,2,3]
            B2 = [1,2,4]
            events2, h2 = Flowfusion.sample_alignment_ud(rng, A2, B2, KUD)
            @test h2 == 0.0
        end
        
        @testset "Grouped vs full-tensor Doob rates agree" begin
            rng = Flowfusion.MersenneTwister(2)
            K = 7
            p = UniformDiscretePoissonIndelProcess(0.9, 0.8, 0.5, K)
            t = 0.4
        
            # small random case
            Xt = rand(rng, 1:K, 8)
            x1 = rand(rng, 1:K, 6)
        
            G = Flowfusion.doob_ud_grouped_fast(p, Xt, x1, t)
            F = Flowfusion.doob_ud_full_tensors_fast(p, Xt, x1, t)
        
            # Reconstruct full tensors from grouped
            K_, n = p.k, length(Xt)
            toks = G.sub_match_tokens
            sub_from_grouped = zeros(eltype(F.sub), K_, n)
            for i in 1:n
            per = (G.sub_other_count[i] > 0) ? G.sub_other[i] / G.sub_other_count[i] : zero(eltype(F.sub))
            @views sub_from_grouped[:, i] .= per
            @views sub_from_grouped[toks, i] .= G.sub_match[:, i]
            sub_from_grouped[Xt[i], i] = 0
            end
        
            ins_from_grouped = zeros(eltype(F.ins), K_, n+1)
            per_ins_other = (G.ins_other_count > 0) ? (G.ins_other ./ G.ins_other_count) : zero(eltype(F.ins))
            for s in 1:(n+1)
            @views ins_from_grouped[:, s] .= per_ins_other[s]
            @views ins_from_grouped[toks, s] .= G.ins_match[:, s]
            end
        
            @test isapprox(F.hcur, G.hcur; atol=1e-10, rtol=1e-10)
            @test maximum(abs.(F.del .- G.del)) ≤ 1e-8
            @test maximum(abs.(F.sub .- sub_from_grouped)) ≤ 1e-8
            @test maximum(abs.(F.ins .- ins_from_grouped)) ≤ 1e-8
        end
        
        for prex0 in [[1,2,4], [1,4], [1]], prex1 in [[1,2,4], [1,4], [1]]
            @testset "Bridge vs Step marginal consistency $prex0 -> $prex1" begin
                Flowfusion.Random.seed!(1234)
                K = 6
                p = UniformDiscretePoissonIndelProcess(0.05, 0.05, 0.05, K)
                x0 = DiscreteState(K, prex0)
                x1 = DiscreteState(K, prex1)
                times = [0.2, 0.5, 0.8]

                # bridge marginals
                Nbridge = 10_000
                bridge_counts = Dict{Float64, Dict{Vector{Int}, Int}}()
                for t in times
                    bridge_counts[t] = Dict{Vector{Int}, Int}()
                    C = Flowfusion.make_precomp(p, t)
                    for _ in 1:Nbridge
                        xt, _ = Flowfusion.sample_Xt_ud(p, tensor(x0), tensor(x1), C)
                        get!(bridge_counts[t], xt, 0)
                        bridge_counts[t][xt] += 1
                    end
                end

                # step marginals via small steps, recording at times
                Nstep = 5_000
                dt = 0.01
                grid = collect(0.0:dt:1.0)
                function record!(d::Dict{Vector{Int}, Int}, x::Vector{Int})
                    get!(d, x, 0); d[x] += 1
                end
                step_counts = Dict(t => Dict{Vector{Int}, Int}() for t in times)
                for _ in 1:Nstep
                    Xt = x0
                    for k in 1:(length(grid)-1)
                        s1, s2 = grid[k], grid[k+1]
                        rates = Flowfusion.doob_ud_full_tensors_fast(p, tensor(Xt), tensor(x1), s1)
                        guide = Flowfusion.Guide((sub=rates.sub, del=rates.del, ins=rates.ins))
                        Xt = Flowfusion.step(p, Xt, guide, s1, s2)
                        if any(abs.(s2 .- times) .< 1e-8)
                            record!(step_counts[s2], tensor(Xt))
                        end
                    end
                end

                # compare via total variation distance
                function tvd(d1::Dict{Vector{Int}, Int}, n1::Int, d2::Dict{Vector{Int}, Int}, n2::Int)
                    keys_all = union(keys(d1), keys(d2))
                    s = 0.0
                    for k in keys_all
                        p = get(d1, k, 0) / n1
                        q = get(d2, k, 0) / n2
                        s += abs(p - q)
                    end
                    return 0.5 * s
                end

                for t in times
                    tv = tvd(bridge_counts[t], Nbridge, step_counts[t], Nstep)
                    @show prex0, prex1, t, tv
                    @test tv ≤ 0.25
                end
            end
        end
    end

    @testset "ModPIP" begin
        @testset "Exact targets agree with sampled alignments" begin
            p = ModPIPProcess(6; lambda = 0.3, mu = 0.3)
            v = validate_targets_vs_sampling(p, [1, 1, 2, 3], [1, 4, 3, 5], 0.4; N = 2_000, seed = 1)
            @test v.del_max ≤ 0.02
            @test v.sub_max ≤ 0.02
            @test v.insc_max ≤ 0.02
            @test v.ins_max ≤ 0.15
        end

        for prex0 in ([1], [1, 2], [1, 2, 4]), prex1 in ([1], [2], [1, 2], [1, 4])
            @testset "Bridge vs Step marginal consistency $prex0 -> $prex1" begin
                p = ModPIPProcess(6; lambda = 0.05, mu = 0.05)
                x0 = DiscreteState(6, prex0)
                x1 = DiscreteState(6, prex1)
                tv = validate_bridge_step_consistency(p, x0, x1; times = [0.2, 0.5, 0.8], Nbridge = 300, Nstep = 300, dt = 0.01, seed = 1)
                @test maximum(values(tv)) ≤ 0.18
            end
        end
    end
end
