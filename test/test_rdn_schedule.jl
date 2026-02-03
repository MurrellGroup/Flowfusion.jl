using Test
using Flowfusion
using ForwardBackward

# Simple mean implementation (avoid Statistics dependency)
_test_mean(x; dims) = sum(x; dims=dims) ./ size(x, dims...)

@testset "RDNFlow Schedule Consistency" begin

    @testset "Schedule transform functions" begin
        # Test that schedules are correct at boundaries
        for schedule in [:linear, :power, :log]
            P = RDNFlow(3; schedule=schedule, schedule_param=2.0f0)

            # At u=0, τ should be 0
            @test schedule_transform(P, 0.0f0) ≈ 0.0f0 atol=1e-6

            # At u=1, τ should be 1
            @test schedule_transform(P, 1.0f0) ≈ 1.0f0 atol=1e-6
        end

        # Test power schedule: τ(u) = u^p
        P_power = RDNFlow(3; schedule=:power, schedule_param=2.0f0)
        @test schedule_transform(P_power, 0.5f0) ≈ 0.25f0 atol=1e-6  # 0.5^2 = 0.25

        # Test log schedule reaches high values quickly
        P_log = RDNFlow(3; schedule=:log, schedule_param=2.0f0)
        tau_log_half = schedule_transform(P_log, 0.5f0)
        @test tau_log_half > 0.8  # Log schedule should be > 0.8 at u=0.5

        # Test that log > power at u=0.5 (log is "faster")
        tau_power_half = schedule_transform(P_power, 0.5f0)
        @test tau_log_half > tau_power_half
    end

    @testset "Bridge vs Step consistency - Linear schedule" begin
        # For linear schedule, bridge and step should give identical results
        P = RDNFlow(3; schedule=:linear, zero_com=false)

        d, n, b = 3, 10, 4
        X0 = ContinuousState(randn(Float32, d, n, b))
        X1 = ContinuousState(randn(Float32, d, n, b))

        for u_target in [0.05f0, 0.25f0, 0.5f0, 0.75f0, 0.95f0]
            # Bridge directly to u_target
            Xt_bridge = bridge(P, X0, X1, u_target)

            # Step from 0 to u_target in small increments
            nsteps = 100
            us = range(0.0f0, u_target, length=nsteps+1)
            Xt_step = X0
            for i in 1:nsteps
                # For ODE step, we need X1 prediction which is just the true X1
                Xt_step = Flowfusion.step(P, Xt_step, X1, us[i], us[i+1])
            end

            # They should match closely (small numerical error from many steps)
            @test isapprox(tensor(Xt_bridge), tensor(Xt_step), rtol=0.01)
        end
    end

    @testset "Bridge vs Step consistency - Power schedule" begin
        # For power schedule, bridge and step should give same distribution
        P = RDNFlow(3; schedule=:power, schedule_param=2.0f0, zero_com=false)

        d, n, b = 3, 10, 4
        X0 = ContinuousState(randn(Float32, d, n, b))
        X1 = ContinuousState(randn(Float32, d, n, b))

        for u_target in [0.05f0, 0.25f0, 0.5f0, 0.75f0, 0.95f0]
            # Bridge directly to u_target
            Xt_bridge = bridge(P, X0, X1, u_target)

            # Step from 0 to u_target in small increments
            nsteps = 100
            us = range(0.0f0, u_target, length=nsteps+1)
            Xt_step = X0
            for i in 1:nsteps
                Xt_step = Flowfusion.step(P, Xt_step, X1, us[i], us[i+1])
            end

            # They should match closely
            @test isapprox(tensor(Xt_bridge), tensor(Xt_step), rtol=0.01)
        end
    end

    @testset "Bridge vs Step consistency - Log schedule" begin
        # For log schedule, bridge and step should give same distribution
        P = RDNFlow(3; schedule=:log, schedule_param=2.0f0, zero_com=false)

        d, n, b = 3, 10, 4
        X0 = ContinuousState(randn(Float32, d, n, b))
        X1 = ContinuousState(randn(Float32, d, n, b))

        for u_target in [0.05f0, 0.25f0, 0.5f0, 0.75f0, 0.95f0]
            # Bridge directly to u_target
            Xt_bridge = bridge(P, X0, X1, u_target)

            # Step from 0 to u_target in small increments
            nsteps = 100
            us = range(0.0f0, u_target, length=nsteps+1)
            Xt_step = X0
            for i in 1:nsteps
                Xt_step = Flowfusion.step(P, Xt_step, X1, us[i], us[i+1])
            end

            # They should match closely
            @test isapprox(tensor(Xt_bridge), tensor(Xt_step), rtol=0.01)
        end
    end

    @testset "Bridge vs Step consistency - with zero_com" begin
        # Test with zero center of mass constraint
        P = RDNFlow(3; schedule=:power, schedule_param=2.0f0, zero_com=true)

        d, n, b = 3, 10, 4
        X0 = ContinuousState(randn(Float32, d, n, b))
        X1 = ContinuousState(randn(Float32, d, n, b))

        for u_target in [0.25f0, 0.5f0, 0.75f0]
            # Bridge directly to u_target
            Xt_bridge = bridge(P, X0, X1, u_target)

            # Step from 0 to u_target in small increments
            nsteps = 100
            us = range(0.0f0, u_target, length=nsteps+1)
            Xt_step = X0
            for i in 1:nsteps
                Xt_step = Flowfusion.step(P, Xt_step, X1, us[i], us[i+1])
            end

            # They should match closely
            @test isapprox(tensor(Xt_bridge), tensor(Xt_step), rtol=0.02)

            # Both should have approximately zero COM
            com_bridge = _test_mean(tensor(Xt_bridge), dims=2)
            com_step = _test_mean(tensor(Xt_step), dims=2)
            @test maximum(abs.(com_bridge)) < 1e-5
            @test maximum(abs.(com_step)) < 1e-5
        end
    end

    @testset "Schedule affects interpolation position" begin
        # Verify that different schedules give different interpolation results
        P_power = RDNFlow(3; schedule=:power, schedule_param=2.0f0, zero_com=false)
        P_log = RDNFlow(3; schedule=:log, schedule_param=2.0f0, zero_com=false)
        P_linear = RDNFlow(3; schedule=:linear, zero_com=false)

        d, n, b = 3, 10, 1
        X0 = ContinuousState(zeros(Float32, d, n, b))
        X1 = ContinuousState(ones(Float32, d, n, b))

        u = 0.5f0
        Xt_power = tensor(bridge(P_power, X0, X1, u))
        Xt_log = tensor(bridge(P_log, X0, X1, u))
        Xt_linear = tensor(bridge(P_linear, X0, X1, u))

        # Power at u=0.5: τ = 0.25, so Xt = 0.25 * X1 = 0.25
        @test all(Xt_power .≈ 0.25f0)

        # Linear at u=0.5: τ = 0.5, so Xt = 0.5 * X1 = 0.5
        @test all(Xt_linear .≈ 0.5f0)

        # Log at u=0.5: τ ≈ 0.909, so Xt ≈ 0.909
        @test all(Xt_log .> 0.9f0)

        # Log should be ahead of linear, which is ahead of power
        @test sum(Xt_log)/length(Xt_log) > sum(Xt_linear)/length(Xt_linear) > sum(Xt_power)/length(Xt_power)
    end

end
