using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
using Flowfusion
const FF = Flowfusion
using Test

@testset "EditFlow remaining_edits, transition mask, remove/pad, loss" begin
    
    tokens = 21
    pad = 22
    lat = 23
    bos = 0
    P = Flowfusion.EditFlow(tokens; transform=identity, padding_token=pad, latent_token=lat, bos_token=bos=0)

    # ─────────────────────────────────────────────────────────────────────
    # remaining_edits: simple 1-column case
    Zt = [0; 7; 23; 23; 4; 23; 22;;]
    Z1 = [0; 7; 20; 20; 4; 10; 22;;]
    expected = zeros(Float32, 2*tokens+1, 4, 1)
    expected[20,2,1] = 2
    expected[10,3,1] = 1
    got = FF.remaining_edits(P, Zt, Z1)

    @test got == expected

    # Two-column case with inserts/subs
    Zt = [0 0; 7 23; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 6, 2)
    expected[5, 4, 1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+10,5,2] = 1
    expected[tokens+20, 4, 1] = 1
    expected[tokens+20, 3, 2] = 1
    expected[tokens+20, 3, 1] = 1
    expected[tokens+20, 2, 2] = 1
    expected[7, 1, 2] = 1
    got = FF.remaining_edits(P, Zt, Z1)
    @test got == expected

    # deletions case
    Zt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;]
    Z1 = [0 0; 7 7; 23 23; 20 23; 4 4; 23 23; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 7, 2)
    expected[2*tokens+1,3,1] = 1
    expected[2*tokens+1,6,1] = 1
    expected[2*tokens+1,3,2] = 1
    expected[2*tokens+1,4,2] = 1
    expected[2*tokens+1,6,2] = 1
    got = FF.remaining_edits(P, Zt, Z1)
    @test got == expected

    # mixed inserts/subs case
    Zt = [0 0; 7 7; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 7, 2)
    expected[5, 4, 1] = 1
    expected[tokens+20,3,1] = 1
    expected[tokens+20,4,1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+20,3,2] = 1
    expected[tokens+20,4,2] = 1
    expected[tokens+10,6,2] = 1
    got = FF.remaining_edits(P, Zt, Z1)
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # transition mask from Xt
    Xt = [0 0; 7 7; 20 20; 20 20; 5 4; 10 22; 22 22;]
    expected = ones(Float32, 2*tokens+1, size(Xt)...)
    # sample 1 (no self-sub mask for BOS=0)
    expected[tokens+7,2,1] = 0
    expected[tokens+20,3,1] = 0
    expected[tokens+20,4,1] = 0
    expected[tokens+5,5,1] = 0
    expected[tokens+10,6,1] = 0
    expected[:,7,1] .= 0
    expected[tokens+1:2*tokens+1,1,1] .= 0
    # sample 2 (no self-sub mask for BOS=0)
    expected[tokens+7,2,2] = 0
    expected[tokens+20,3,2] = 0
    expected[tokens+20,4,2] = 0
    expected[tokens+4,5,2] = 0
    expected[:,6:7,2] .= 0
    expected[tokens+1:2*tokens+1,1,2] .= 0


    got = FF.transition_mask_from_Xt(P, Xt)

    # Display all indices in `got` where the value is zero
    zero_indices = findall(x -> x == 0, got)
    #@info "Indices in 'got' that are zero:" zero_indices
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # loss equivalence under identity transform
    edit_multiplier = [0; 2;; 1; 0;;; 1; 0;; 0; 1;;;]     # (2,2,2)
    transition_mask = [1; 0;; 1; 0;;; 1; 0;; 0; 1;;;]     # (2,2,2)
    M = [0.1; 0.2;; 0.3; 0.4;;; 0.5; 0.6;; 0.7; 0.8;;;]   # (2,2,2)
    t = [0.3; 0.7;;]                                       # (1,2)
    k(t)=t; dk(t)=1
    scheduler_scaling = dk.(t) ./ (-k.(t) .+ 1)
    # manual loss
    l = (0.1+0.3+0.5+0.8 - (1/(1-0.3)*(2*log(0.2)+log(0.3)) + 1/(1-0.7)*(log(0.5)+log(0.8))))/2
    got = Flowfusion.edit_loss(P, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=0)
    @test isapprox(got, l; atol=1e-7, rtol=1e-7)

end


