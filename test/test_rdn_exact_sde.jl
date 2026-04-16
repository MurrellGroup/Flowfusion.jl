using Statistics
using Random

@testset "RDNExactSDEFlow" begin
    Random.seed!(1234)

    function _flat_stats(X)
        vals = vec(Array(tensor(X)))
        return (mean(vals), var(vals))
    end

    function _check_marginal_match(P_exact, P_det, u; nsamples = 6000, x1_value = 0.7)
        X0 = ContinuousState(randn(Float64, 1, 1, nsamples))
        X1 = ContinuousState(fill(x1_value, 1, 1, nsamples))

        Xt_det = bridge(P_det, X0, X1, u)
        Xt_sde = bridge(P_exact, X0, X1, u)

        mean_det, var_det = _flat_stats(Xt_det)
        mean_sde, var_sde = _flat_stats(Xt_sde)

        @test isapprox(mean_sde, mean_det; atol = 5e-2)
        @test isapprox(var_sde, var_det; atol = 5e-2)
    end

    function _check_step_match(P_exact, u1, u2; nsamples = 6000, x1_value = -0.3)
        X0 = ContinuousState(randn(Float64, 1, 1, nsamples))
        X1 = ContinuousState(fill(x1_value, 1, 1, nsamples))

        Xmid = bridge(P_exact, X0, X1, u1)
        Xstep = Flowfusion.step(P_exact, Xmid, X1, u1, u2)
        Xdirect = bridge(P_exact, X0, X1, u2)

        mean_step, var_step = _flat_stats(Xstep)
        mean_direct, var_direct = _flat_stats(Xdirect)

        @test isapprox(mean_step, mean_direct; atol = 5e-2)
        @test isapprox(var_step, var_direct; atol = 5e-2)
    end

    @testset "1/t marginal matches deterministic bridge" begin
        P_det = RDNFlow(
            1;
            zero_com = false,
            schedule = :log,
            schedule_param = 2.0,
            sde_gt_mode = Symbol("1/t"),
        )
        P_exact = RDNExactSDEFlow(
            1;
            zero_com = false,
            schedule = :log,
            schedule_param = 2.0,
            sde_gt_mode = Symbol("1/t"),
            sc_scale_noise = 1.0,
        )
        _check_marginal_match(P_exact, P_det, 0.37)
        _check_step_match(P_exact, 0.23, 0.81)
    end

    @testset "tan marginal matches deterministic bridge" begin
        P_det = RDNFlow(
            1;
            zero_com = false,
            schedule = :power,
            schedule_param = 2.0,
            sde_gt_mode = :tan,
        )
        P_exact = RDNExactSDEFlow(
            1;
            zero_com = false,
            schedule = :power,
            schedule_param = 2.0,
            sde_gt_mode = :tan,
            sc_scale_noise = 1.0,
        )
        _check_marginal_match(P_exact, P_det, 0.63)
        _check_step_match(P_exact, 0.28, 0.74)
    end

    @testset "Bridge hits endpoint exactly" begin
        P_exact = RDNExactSDEFlow(
            1;
            zero_com = false,
            schedule = :log,
            schedule_param = 2.0,
            sde_gt_mode = Symbol("1/t"),
            sc_scale_noise = 1.0,
        )
        X0 = ContinuousState(randn(Float64, 1, 1, 32))
        X1 = ContinuousState(randn(Float64, 1, 1, 32))
        Xt = bridge(P_exact, X0, X1, 1.0)
        @test isapprox(tensor(Xt), tensor(X1))
    end
end
