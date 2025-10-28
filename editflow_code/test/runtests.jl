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
    Xt = [0; 7; 4; 22;;]
    Z1 = [0; 7; 20; 20; 4; 10; 22;;]
    expected = zeros(Float32, 2*tokens+1, size(Xt)...)
    expected[20,2,1] = 2
    expected[10,3,1] = 1
    got = FF.remaining_edits(P, Zt, Z1, Xt)

    @test got == expected
    # Two-column case with inserts/subs
    Zt = [0 0; 7 23; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Xt = [0 0; 7 15; 15 15; 15 4; 2 2]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt)...)
    expected[5, 4, 1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+10,5,2] = 1
    expected[tokens+20, 4, 1] = 1
    expected[tokens+20, 3, 2] = 1
    expected[tokens+20, 3, 1] = 1
    expected[tokens+20, 2, 2] = 1
    expected[7, 1, 2] = 1
    got = FF.remaining_edits(P, Zt, Z1, Xt)
    @test got == expected

    # deletions case
    Zt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;]
    Xt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;]
    Z1 = [0 0; 7 7; 23 23; 20 23; 4 4; 23 23; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt)...)
    expected[2*tokens+1,3,1] = 1
    expected[2*tokens+1,6,1] = 1
    expected[2*tokens+1,3,2] = 1
    expected[2*tokens+1,4,2] = 1
    expected[2*tokens+1,6,2] = 1
    got = FF.remaining_edits(P, Zt, Z1, Xt)
    @test got == expected

    # mixed inserts/subs case
    Zt = [0 0; 7 7; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Xt = [0 0; 7 7; 15 15; 15 15; 2 4; 22 2;]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt)...)
    expected[5, 4, 1] = 1
    expected[tokens+20,3,1] = 1
    expected[tokens+20,4,1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+20,3,2] = 1
    expected[tokens+20,4,2] = 1
    expected[tokens+10,6,2] = 1
    got = FF.remaining_edits(P, Zt, Z1, Xt)
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


@testset "EditFlow gap-wise inserts: remaining_edits, transition mask, loss" begin
    tokens = 21
    pad = 22
    lat = 23
    bos = 0
    P = Flowfusion.EditFlow(tokens; transform=identity, padding_token=pad, latent_token=lat, bos_token=bos)

    # ─────────────────────────────────────────────────────────────────────
    # remaining_edits: simple 1-column case (gap-wise)
    # Xt has L=4 rows → gaps = 1..4
    Zt = [0; 7; 23; 23; 4; 23; 22;;]
    Xt = [0; 7; 4; 22;;]
    Z1 = [0; 7; 20; 20; 4; 10; 22;;]
    expected = zeros(Float32, 2*tokens+1, size(Xt,1)+1, size(Xt,2))  # (2K+1, L+1=5? careful: size(Xt,1)=4 -> L+1=5 ; but real Lb=3 → valid gaps 1..4)
    # Inserts are on gaps:
    # - two '20' inserted between 7 and 4 -> gap 3
    # - one '10' inserted after 4 -> gap 4
    expected[20, 3, 1] = 2
    expected[10, 4, 1] = 1
    got = FF.remaining_edits_gapwise(P, Zt, Z1, Xt)
    @test got == expected

    # Two-column case with inserts/subs (gap-wise inserts)
    Zt = [0 0; 7 23; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Xt = [0 0; 7 15; 15 15; 15 4; 2 2]           # L=5 → gaps 1..6
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt,1)+1, size(Xt,2))  # (2K+1, 6, 2)

    # sample 1 (b=1):
    # subs: 15→20 at sites 3 & 4,  2→10 at site 5
    expected[tokens+20, 3, 1] = 1
    expected[tokens+20, 4, 1] = 1
    expected[tokens+10, 5, 1] = 1
    # insert: LAT→5 occurs after site 4 (between 15 and 2) → gap 5
    expected[5, 5, 1] = 1

    # sample 2 (b=2):
    # insert: LAT→7 occurs after site 1 (between BOS and first real) → gap 2
    expected[7, 2, 2] = 1
    # subs: 15→20 at sites 2 & 3,  2→10 at site 5
    expected[tokens+20, 2, 2] = 1
    expected[tokens+20, 3, 2] = 1
    expected[tokens+10, 5, 2] = 1

    got = FF.remaining_edits_gapwise(P, Zt, Z1, Xt)
    @test got == expected

    # deletions case (gap-wise array but deletions live on sites)
    Zt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;]
    Xt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;] # L=7 → gaps 1..8
    Z1 = [0 0; 7 7; 23 23; 20 23; 4 4; 23 23; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt,1)+1, size(Xt,2))

    # sample 1 deletions at sites 3 and 6
    expected[2*tokens+1, 3, 1] = 1
    expected[2*tokens+1, 6, 1] = 1
    # sample 2 deletions at sites 3,4,6
    expected[2*tokens+1, 3, 2] = 1
    expected[2*tokens+1, 4, 2] = 1
    expected[2*tokens+1, 6, 2] = 1

    got = FF.remaining_edits_gapwise(P, Zt, Z1, Xt)
    @test got == expected

    # mixed inserts/subs case (gap-wise)
    Zt = [0 0; 7 7; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Xt = [0 0; 7 7; 15 15; 15 15; 2 4; 22 2;]   # L=6 → gaps 1..7
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, size(Xt,1)+1, size(Xt,2))

    # sample 1: subs at sites 3,4 and 5; insert '5' at gap 5
    expected[tokens+20, 3, 1] = 1
    expected[tokens+20, 4, 1] = 1
    expected[tokens+10, 5, 1] = 1
    expected[5, 5, 1] = 1

    # sample 2: subs at sites 3,4 and 6
    expected[tokens+20, 3, 2] = 1
    expected[tokens+20, 4, 2] = 1
    expected[tokens+10, 6, 2] = 1

    got = FF.remaining_edits_gapwise(P, Zt, Z1, Xt)
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # transition mask from Xt (gap-wise)
    Xt = [0 0; 7 7; 20 20; 20 20; 5 4; 10 22; 22 22;] # L=7 → gaps 1..8
    expected = ones(Float32, 2*tokens+1, size(Xt,1)+1, size(Xt,2))

    # Helper to apply rules programmatically (mirrors gap-wise mask fn)
    function fill_expected_mask!(E, x::Vector{Int}, b::Int)
        L = length(x)
        Lb = count(!=(pad), x)

        # sub/del never valid at last column (gap L+1)
        E[(tokens+1):(2*tokens+1), L+1, b] .= 0

        # pre-BOS insert forbidden
        E[1:tokens, 1, b] .= 0

        # padding sites off
        if Lb < L
            E[:, (Lb+1):L, b] .= 0
        end
        # inserts beyond gap (Lb+1) off
        if (Lb+1) < (L+1)
            E[1:tokens, (Lb+2):(L+1), b] .= 0
        end

        for i in 1:Lb
            t = x[i]
            if t == bos
                # block all subs at BOS and delete BOS
                E[(tokens+1):(2*tokens), i, b] .= 0
                E[2*tokens+1, i, b] = 0
            else
                # self-sub off
                E[tokens + t, i, b] = 0
            end
        end
        return E
    end

    expected = fill_expected_mask!(expected, Xt[:,1], 1)
    expected = fill_expected_mask!(expected, Xt[:,2], 2)

    got = FF.transition_mask_from_Xt_gapwise(P, Xt)
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # loss equivalence under identity transform — with L+1 axis
    # Make tiny tensors of shape (C=2, L+1=3, B=2); third column is masked out.
    edit_multiplier = reshape(Float32[
        0, 2, 0,   1, 0, 0,   # batch 1 (cols 1..3)
        1, 0, 0,   0, 1, 0    # batch 2
    ], (2, 3, 2))

    transition_mask = reshape(Float32[
        1, 0, 0,   1, 0, 0,
        1, 0, 0,   0, 1, 0
    ], (2, 3, 2))

    M = reshape(Float32[
        0.1, 0.2, 0.0,   0.3, 0.4, 0.0,
        0.5, 0.6, 0.0,   0.7, 0.8, 0.0
    ], (2, 3, 2))

    t = reshape(Float32[0.3, 0.7], (1, 2))  # per-batch scheduler t
    k(t)=t; dk(t)=1
    scheduler_scaling = dk.(t) ./ (1 .- k.(t))

    # Manual loss matches your earlier numbers (3rd col masked out)
    l = (0.1+0.3+0.5+0.8 - (1/(1-0.3)*(2*log(0.2)+log(0.3)) + 1/(1-0.7)*(log(0.5)+log(0.8))))/2
    got = Flowfusion.edit_loss(P, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=1e-9)
    @test isapprox(got, l; atol=1e-7, rtol=1e-7)
end
