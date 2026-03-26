module ModPIP

import ..Flowfusion
using ForwardBackward
using Random
using NNlib
using Distributions
import ..Flowfusion: bridge, step, Guide, floss

export
    ModPIPProcess,
    bridge_with_targets,
    exact_targets,
    validate_targets_vs_sampling,
    validate_bridge_step_consistency


"""
    ModPIPProcess(k; lambda=1.0, mu=1.0, kappa, dkappa, count_transform=NNlib.softplus)

PIP-like variable-length bridge with:
- structural alignment from a Poisson indel process
- DFM `x1` prediction on matched-site token content
- non-AR insertion targets given by future token histograms per gap

This file is included into `Flowfusion` as `Flowfusion.ModPIP`.
"""
struct ModPIPProcess{T,F,G,H,D} <: Flowfusion.DiscreteIndelProcess
    lambda::T
    mu::T
    k::Int
    kappa::F
    dkappa::G
    count_transform::H
    dummy_token::D
end

ModPIPProcess(k; lambda=1.0, mu=1.0,
              kappa = t -> 1 - cos((pi / 2) * t),
              dkappa = t -> (pi / 2) * sin((pi / 2) * t),
              count_transform = NNlib.softplus,
              dummy_token = nothing) =
    ModPIPProcess(lambda, mu, k, kappa, dkappa, count_transform, dummy_token)


struct AlignWeights{T}
    del::T
    ins::T
    mat_same::T
    mat_other::T
    mat_dummy::T
    dummy_token::Int
end

struct AlignEvent
    typ::Symbol
    i::Int
    j::Int
end


@inline function delete_weight(W::AlignWeights{T}, a::Int) where {T}
    if W.dummy_token != 0 && a != W.dummy_token
        return zero(T)
    end
    return W.del
end


@inline function logaddexp(a::T, b::T) where {T}
    if a == -Inf
        return b
    elseif b == -Inf
        return a
    elseif b > a
        a, b = b, a
    end
    return a + log1p(exp(b - a))
end

@inline mod_survival(mu, s) = exp(-mu * s)

@inline function mod_integrated_survival(mu, s)
    abs(mu) < 1e-8 && return s
    return -expm1(-mu * s) / mu
end

@inline mod_insert_mass(p::ModPIPProcess, s) = p.lambda * mod_integrated_survival(p.mu, s)

@inline function full_weights(p::ModPIPProcess{T}) where {T}
    surv = mod_survival(p.mu, one(T))
    # Full x0-x1 bridge alignment stays structurally blind: any matched token pair is allowed.
    AlignWeights{T}(one(T) - surv, mod_insert_mass(p, one(T)), surv, surv, surv, 0)
end

@inline function remaining_weights(p::ModPIPProcess{T}, t::T) where {T}
    s = one(T) - t
    surv = mod_survival(p.mu, s)
    kap = p.kappa(t)
    if isnothing(p.dummy_token)
        # Uniform-AA x0 prior: matched current token can be the target token
        # or any source token with equal prior mass.
        off = (one(T) - kap) / p.k
        same = kap + off
        return AlignWeights{T}(one(T) - surv, mod_insert_mass(p, s), surv * same, surv * off, zero(T), 0)
    else
        # Mask-token x0 prior: matched current token can only be the mask token
        # or the target token.
        return AlignWeights{T}(one(T) - surv, mod_insert_mass(p, s), surv * kap, zero(T), surv * (one(T) - kap), Int(p.dummy_token))
    end
end

@inline function delete_alive_prob(mu, t)
    if abs(mu) < 1e-8
        return max(zero(t), one(t) - t)
    end
    denom = one(t) - exp(-mu)
    num = exp(-mu * t) - exp(-mu)
    return clamp(num / max(eps(t), denom), zero(t), one(t))
end

@inline function insert_present_prob(mu, t)
    if abs(mu) < 1e-8
        return clamp(t, zero(t), one(t))
    end
    denom = one(t) - exp(-mu)
    num = exp(-mu * (one(t) - t)) - exp(-mu)
    return clamp(num / max(eps(t), denom), zero(t), one(t))
end

@inline function delete_hazard(mu, t)
    if abs(mu) < 1e-8
        return inv(max(eps(t), one(t) - t))
    end
    tail = one(t) - exp(-mu * (one(t) - t))
    return mu / max(eps(t), tail)
end

@inline function insert_hazard(mu, t)
    if abs(mu) < 1e-8
        return inv(max(eps(t), one(t) - t))
    end
    e = exp(-mu * (one(t) - t))
    return mu * e / max(eps(t), one(t) - e)
end

@inline function sub_hazard(p::ModPIPProcess, t)
    kap = p.kappa(t)
    return p.dkappa(t) / max(eps(typeof(kap)), one(kap) - kap)
end


@inline function match_weight(W::AlignWeights{T}, a::Int, b::Int) where {T}
    if a == b
        return W.mat_same
    elseif W.dummy_token != 0 && a == W.dummy_token
        return W.mat_dummy
    else
        return W.mat_other
    end
end

function forward_dp!(F::AbstractMatrix{T}, A::AbstractVector{Int}, B::AbstractVector{Int}, W::AlignWeights{T}) where {T}
    n, m = length(A), length(B)
    F[1:n+1, 1:m+1] .= zero(T)
    F[1, 1] = one(T)
    for i in 0:n, j in 0:m
        f = F[i + 1, j + 1]
        if i < n
            F[i + 2, j + 1] += f * delete_weight(W, A[i + 1])
        end
        if j < m
            F[i + 1, j + 2] += f * W.ins
        end
        if i < n && j < m
            F[i + 2, j + 2] += f * match_weight(W, A[i + 1], B[j + 1])
        end
    end
    return F[n + 1, m + 1]
end

function sample_alignment(rng::AbstractRNG, A::AbstractVector{Int}, B::AbstractVector{Int}, W::AlignWeights{T}) where {T}
    n, m = length(A), length(B)
    F = zeros(T, n + 1, m + 1)
    h = forward_dp!(F, A, B, W)

    events = AlignEvent[]
    i, j = n, m
    while i > 0 || j > 0
        wdel = i > 0 ? F[i, j + 1] * delete_weight(W, A[i]) : zero(T)
        wins = j > 0 ? F[i + 1, j] * W.ins : zero(T)
        wmat = (i > 0 && j > 0) ? F[i, j] * match_weight(W, A[i], B[j]) : zero(T)
        wtot = wdel + wins + wmat

        if !(wtot > 0)
            if i > 0 && j > 0
                push!(events, AlignEvent(:R, i, j))
                i -= 1
                j -= 1
            elseif i > 0
                push!(events, AlignEvent(:A, i, 0))
                i -= 1
            else
                push!(events, AlignEvent(:B, 0, j))
                j -= 1
            end
            continue
        end

        u = rand(rng) * wtot
        if u < wdel
            push!(events, AlignEvent(:A, i, 0))
            i -= 1
        elseif u < wdel + wins
            push!(events, AlignEvent(:B, 0, j))
            j -= 1
        else
            push!(events, AlignEvent(:R, i, j))
            i -= 1
            j -= 1
        end
    end
    reverse!(events)
    return events, h
end


function dp_logs!(logF::AbstractMatrix{T},
                  logB::AbstractMatrix{T},
                  A::AbstractVector{Int},
                  B::AbstractVector{Int},
                  W::AlignWeights{T}) where {T}
    n, m = length(A), length(B)
    fill!(logF, -Inf)
    fill!(logB, -Inf)
    logF[1, 1] = zero(T)

    log_ins = W.ins > 0 ? log(W.ins) : T(-Inf)
    log_mat_same = W.mat_same > 0 ? log(W.mat_same) : T(-Inf)
    log_mat_other = W.mat_other > 0 ? log(W.mat_other) : T(-Inf)
    log_mat_dummy = W.mat_dummy > 0 ? log(W.mat_dummy) : T(-Inf)

    for i in 0:n, j in 0:m
        f = logF[i + 1, j + 1]
        f == -Inf && continue
        if i < n
            delw = delete_weight(W, A[i + 1])
            log_del = delw > 0 ? log(delw) : T(-Inf)
            logF[i + 2, j + 1] = logaddexp(logF[i + 2, j + 1], f + log_del)
        end
        if j < m
            logF[i + 1, j + 2] = logaddexp(logF[i + 1, j + 2], f + log_ins)
        end
        if i < n && j < m
            log_mat = A[i + 1] == B[j + 1] ? log_mat_same :
                      (W.dummy_token != 0 && A[i + 1] == W.dummy_token ? log_mat_dummy : log_mat_other)
            logF[i + 2, j + 2] = logaddexp(logF[i + 2, j + 2], f + log_mat)
        end
    end

    logB[n + 1, m + 1] = zero(T)
    for i in n:-1:0, j in m:-1:0
        (i == n && j == m) && continue
        acc = T(-Inf)
        if i < n
            delw = delete_weight(W, A[i + 1])
            log_del = delw > 0 ? log(delw) : T(-Inf)
            acc = logaddexp(acc, log_del + logB[i + 2, j + 1])
        end
        if j < m
            acc = logaddexp(acc, log_ins + logB[i + 1, j + 2])
        end
        if i < n && j < m
            log_mat = A[i + 1] == B[j + 1] ? log_mat_same :
                      (W.dummy_token != 0 && A[i + 1] == W.dummy_token ? log_mat_dummy : log_mat_other)
            acc = logaddexp(acc, log_mat + logB[i + 2, j + 2])
        end
        logB[i + 1, j + 1] = acc
    end

    return logF[n + 1, m + 1], log_ins, log_mat_same, log_mat_other, log_mat_dummy
end


function hard_targets_from_events(k::Int, x1::Vector{Int}, ncur::Int, events::Vector{AlignEvent})
    del = zeros(Float64, 1, ncur)
    sub = zeros(Float64, k, ncur)
    sub_weight = zeros(Float64, 1, ncur)
    ins_hist = zeros(Float64, k, ncur + 1)
    ins_count = zeros(Float64, 1, ncur + 1)
    ins_weight = zeros(Float64, 1, ncur + 1)

    site_idx = 0
    gap_idx = 1
    for ev in events
        if ev.typ === :A
            site_idx += 1
            del[1, site_idx] = 1.0
            gap_idx = site_idx + 1
        elseif ev.typ === :R
            site_idx += 1
            sub[x1[ev.j], site_idx] = 1.0
            sub_weight[1, site_idx] = 1.0
            gap_idx = site_idx + 1
        else
            tok = x1[ev.j]
            ins_hist[tok, gap_idx] += 1.0
            ins_count[1, gap_idx] += 1.0
            ins_weight[1, gap_idx] += 1.0
        end
    end

    ins_tok = zeros(Float64, size(ins_hist))
    for s in 1:(ncur + 1)
        if ins_count[1, s] > 0
            ins_tok[:, s] .= ins_hist[:, s] ./ ins_count[1, s]
        end
    end

    return (del = del,
            sub = sub,
            sub_weight = sub_weight,
            ins_tok = ins_tok,
            ins_count = ins_count,
            ins_weight = ins_weight)
end


function bridge_with_targets(rng::AbstractRNG,
                             p::ModPIPProcess{T},
                             x0::DiscreteState{<:AbstractArray{<:Signed}},
                             x1::DiscreteState{<:AbstractArray{<:Signed}},
                             t::Real) where {T}
    @assert ndims(x0.state) == 1
    @assert ndims(x1.state) == 1

    tT = T(t)
    x0v = collect(Int, tensor(x0))
    x1v = collect(Int, tensor(x1))

    events, _ = sample_alignment(rng, x0v, x1v, full_weights(p))
    palive = delete_alive_prob(p.mu, tT)
    ppresent = insert_present_prob(p.mu, tT)
    kap = p.kappa(tT)

    xt = Int[]
    del = Float32[]
    sub_tok = Int[]
    sub_mask = Bool[]
    gaps = [Int[]]

    for ev in events
        if ev.typ === :A
            if rand(rng) < palive
                push!(xt, x0v[ev.i])
                push!(del, 1f0)
                push!(sub_tok, 0)
                push!(sub_mask, false)
                push!(gaps, Int[])
            end
        elseif ev.typ === :R
            a = x0v[ev.i]
            b = x1v[ev.j]
            tok = (a == b || rand(rng) >= kap) ? a : b
            push!(xt, tok)
            push!(del, 0f0)
            push!(sub_tok, b)
            push!(sub_mask, true)
            push!(gaps, Int[])
        else
            b = x1v[ev.j]
            if rand(rng) < ppresent
                push!(xt, b)
                push!(del, 0f0)
                push!(sub_tok, b)
                push!(sub_mask, true)
                push!(gaps, Int[])
            else
                push!(gaps[end], b)
            end
        end
    end

    ins_count = reshape(Float32.(length.(gaps)), 1, :)
    ins_weight = copy(ins_count)
    ins_tok = zeros(Float32, p.k, length(gaps))
    for s in eachindex(gaps)
        if !isempty(gaps[s])
            for tok in gaps[s]
                ins_tok[tok, s] += 1f0
            end
            ins_tok[:, s] ./= length(gaps[s])
        end
    end

    hard = (del = reshape(del, 1, :),
            sub = let out = zeros(Float32, p.k, length(xt))
                for i in eachindex(sub_tok)
                    sub_mask[i] && (out[sub_tok[i], i] = 1f0)
                end
                out
            end,
            sub_weight = reshape(Float32.(sub_mask), 1, :),
            ins_tok = ins_tok,
            ins_count = ins_count,
            ins_weight = ins_weight,
            gap_queues = gaps,
            sub_tok = sub_tok,
            sub_mask = sub_mask)

    return DiscreteState(x0.K, xt), hard
end

bridge_with_targets(p::ModPIPProcess, x0, x1, t) =
    bridge_with_targets(Random.default_rng(), p, x0, x1, t)


function exact_targets(p::ModPIPProcess{T}, xt::Vector{Int}, x1::Vector{Int}, t::T) where {T}
    n, m = length(xt), length(x1)
    W = remaining_weights(p, t)
    logF = fill(T(-Inf), n + 1, m + 1)
    logB = similar(logF)
    logh, log_ins, log_mat_same, log_mat_other, log_mat_dummy = dp_logs!(logF, logB, xt, x1, W)

    sub_mass = zeros(T, p.k, n)
    del_prob = zeros(T, 1, n)
    sub_weight = zeros(T, 1, n)

    for i in 1:n
        acc_del = T(-Inf)
        delw = delete_weight(W, xt[i])
        log_del = delw > 0 ? log(delw) : T(-Inf)
        for j in 0:m
            acc_del = logaddexp(acc_del, logF[i, j + 1] + log_del + logB[i + 1, j + 1])
        end
        del_prob[1, i] = acc_del == -Inf ? zero(T) : exp(acc_del - logh)

        for j in 1:m
            log_mat = xt[i] == x1[j] ? log_mat_same :
                      (W.dummy_token != 0 && xt[i] == W.dummy_token ? log_mat_dummy : log_mat_other)
            val = logF[i, j] + log_mat + logB[i + 1, j + 1]
            val == -Inf && continue
            sub_mass[x1[j], i] += exp(val - logh)
        end
        sub_weight[1, i] = sum(@view sub_mass[:, i])
        if sub_weight[1, i] > 0
            sub_mass[:, i] ./= sub_weight[1, i]
        end
    end

    ins_mass = zeros(T, p.k, n + 1)
    ins_count = zeros(T, 1, n + 1)
    ins_weight = zeros(T, 1, n + 1)
    for s in 0:n
        for j in 1:m
            val = logF[s + 1, j] + log_ins + logB[s + 1, j + 1]
            val == -Inf && continue
            ins_mass[x1[j], s + 1] += exp(val - logh)
        end
        ins_count[1, s + 1] = sum(@view ins_mass[:, s + 1])
        ins_weight[1, s + 1] = ins_count[1, s + 1]
        if ins_count[1, s + 1] > 0
            ins_mass[:, s + 1] ./= ins_count[1, s + 1]
        end
    end

    return (del = del_prob,
            sub = sub_mass,
            sub_weight = sub_weight,
            ins_tok = ins_mass,
            ins_count = ins_count,
            ins_weight = ins_weight,
            hcur = exp(logh))
end


@inline function valid_token_indices(k::Int, dummy_token)
    if isnothing(dummy_token)
        return collect(1:k)
    end
    d = Int(dummy_token)
    if d == k
        return collect(1:(k - 1))
    elseif d == 1
        return collect(2:k)
    else
        return vcat(collect(1:(d - 1)), collect((d + 1):k))
    end
end

function masked_softmax(logits::AbstractArray, dummy_token)
    isnothing(dummy_token) && return NNlib.softmax(logits)
    valid = valid_token_indices(size(logits, 1), dummy_token)
    probs_valid = if ndims(logits) == 3
        NNlib.softmax(logits[valid, :, :])
    elseif ndims(logits) == 2
        NNlib.softmax(logits[valid, :])
    else
        error("masked_softmax expects a 2D or 3D array, got ndims=$(ndims(logits))")
    end
    out = similar(logits)
    fill!(out, zero(eltype(out)))
    if ndims(logits) == 3
        out[valid, :, :] .= probs_valid
    else
        out[valid, :] .= probs_valid
    end
    return out
end


function Guide(p::ModPIPProcess{T},
               t::Real,
               xt::DiscreteState{<:AbstractArray{<:Signed}},
               x1::DiscreteState{<:AbstractArray{<:Signed}}) where {T}
    @assert ndims(xt.state) == 1
    @assert ndims(x1.state) == 1
    tg = exact_targets(p, collect(Int, tensor(xt)), collect(Int, tensor(x1)), T(t))
    n = length(tensor(xt))
    return Flowfusion.Guide(tg, trues(n + 1), trues(n))
end

function Guide(p::ModPIPProcess{T},
               tvec::AbstractVector{<:Real},
               xts::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
               x1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}}) where {T}
    B = length(xts)
    @assert B == length(x1s) == length(tvec)

    lens = length.(tensor.(xts))
    nmax = maximum(vcat(0, lens))
    K = p.k
    S = eltype(tvec)

    del = zeros(S, 1, nmax, B)
    sub = zeros(S, K, nmax, B)
    sub_weight = zeros(S, 1, nmax, B)
    ins_tok = zeros(S, K, nmax + 1, B)
    ins_count = zeros(S, 1, nmax + 1, B)
    ins_weight = zeros(S, 1, nmax + 1, B)

    lmask = falses(nmax, B)
    gapmask = falses(nmax + 1, B)

    for b in 1:B
        xt = collect(Int, tensor(xts[b]))
        x1 = collect(Int, tensor(x1s[b]))
        n = length(xt)
        tg = exact_targets(p, xt, x1, T(tvec[b]))
        if n > 0
            del[:, 1:n, b] .= S.(tg.del)
            sub[:, 1:n, b] .= S.(tg.sub)
            sub_weight[:, 1:n, b] .= S.(tg.sub_weight)
            lmask[1:n, b] .= true
        end
        ins_tok[:, 1:(n + 1), b] .= S.(tg.ins_tok)
        ins_count[:, 1:(n + 1), b] .= S.(tg.ins_count)
        ins_weight[:, 1:(n + 1), b] .= S.(tg.ins_weight)
        gapmask[1:(n + 1), b] .= true
    end

    return Flowfusion.Guide((del = del,
                             sub = sub,
                             sub_weight = sub_weight,
                             ins_tok = ins_tok,
                             ins_count = ins_count,
                             ins_weight = ins_weight),
                            gapmask,
                            lmask)
end


function bridge(p::ModPIPProcess,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t::Real)
    xt, _ = bridge_with_targets(Random.default_rng(), p, x0, x1, t)
    return xt
end


@inline bce_with_logits(logits, target) = NNlib.softplus.(logits) .- target .* logits

function pos_breg(p::AbstractArray{T}, q::AbstractArray{T}; eps = T(1e-8)) where {T}
    return p .* (log.(p .+ eps) .- log.(q .+ eps)) .- p .+ q
end

function floss(p::ModPIPProcess,
               xt::Flowfusion.MaskedState{<:DiscreteState},
               xhat,
               g::Flowfusion.Guide,
               c)
    # Model contract:
    # - xhat.del      :: (1, n, B) logits for delete-by-1
    # - xhat.sub      :: (K, n, B) logits for final-token distribution
    # - xhat.ins_tok  :: (K, n+1, B) logits for future inserted-token histogram
    # - xhat.ins_count:: (1, n+1, B) unconstrained count parameter
    del_logits = xhat.del
    sub_logits = xhat.sub
    ins_tok_logits = xhat.ins_tok
    ins_count = p.count_transform.(xhat.ins_count)
    valid = valid_token_indices(size(sub_logits, 1), p.dummy_token)

    del_loss = bce_with_logits(del_logits, g.H.del)
    sub_ce = -sum(g.H.sub[valid, :, :] .* NNlib.logsoftmax(sub_logits[valid, :, :]), dims = 1)
    ins_ce = -sum(g.H.ins_tok[valid, :, :] .* NNlib.logsoftmax(ins_tok_logits[valid, :, :]), dims = 1)
    count_loss = pos_breg(g.H.ins_count, ins_count)

    return Flowfusion.scaledmaskedmean(del_loss, c, Flowfusion.getlmask(g)) +
           Flowfusion.scaledmaskedmean(g.H.sub_weight .* sub_ce, c, Flowfusion.getlmask(g)) +
           Flowfusion.scaledmaskedmean(count_loss, c, Flowfusion.getcmask(g)) +
           Flowfusion.scaledmaskedmean(g.H.ins_weight .* ins_ce, c, Flowfusion.getcmask(g))
end


function _unbatch_last(A::AbstractArray)
    ndims(A) == 3 && return Array(A[:, :, 1])
    ndims(A) == 2 && return Array(A)
    error("Expected a 2D or 3D array, got ndims=$(ndims(A))")
end

function step(p::ModPIPProcess,
              xt::DiscreteState{<:AbstractArray{<:Signed}},
              hat,
              s1::Real,
              s2::Real)
    @assert ndims(xt.state) == 1
    guided = hat isa Flowfusion.Guide
    h = guided ? hat.H : hat

    x = collect(Int, tensor(xt))
    n = length(x)
    dt = float(s2 - s1)

    if guided
        del_prob = vec(_unbatch_last(h.del))
        sub_prob = _unbatch_last(h.sub)
        ins_tok_prob = _unbatch_last(h.ins_tok)
        ins_count = vec(_unbatch_last(h.ins_count))
    else
        del_logits = _unbatch_last(h.del)
        sub_logits = _unbatch_last(h.sub)
        ins_tok_logits = _unbatch_last(h.ins_tok)
        ins_count_raw = _unbatch_last(h.ins_count)

        del_prob = vec(NNlib.sigmoid.(del_logits))
        sub_prob = masked_softmax(sub_logits, p.dummy_token)
        ins_tok_prob = masked_softmax(ins_tok_logits, p.dummy_token)
        ins_count = vec(p.count_transform.(ins_count_raw))
    end

    if !isnothing(p.dummy_token)
        dummy = Int(p.dummy_token)
        for i in 1:n
            if x[i] != dummy
                del_prob[i] = zero(eltype(del_prob))
                sub_prob[:, i] .= zero(eltype(sub_prob))
            else
                sub_prob[dummy, i] = zero(eltype(sub_prob))
            end
        end
        ins_tok_prob[dummy, :] .= zero(eltype(ins_tok_prob))
    end

    r_del = delete_hazard(p.mu, s1) .* del_prob
    r_sub = sub_hazard(p, s1) .* (reshape(one(eltype(sub_prob)) .- del_prob, 1, :)) .* sub_prob
    for i in 1:n
        r_sub[x[i], i] = zero(eltype(r_sub))
    end
    r_ins = insert_hazard(p.mu, s1) .* reshape(ins_count, 1, :) .* ins_tok_prob

    to_delete = falses(n)
    sub_to = zeros(Int, n)
    for i in 1:n
        rsub_tot = sum(@view r_sub[:, i])
        rtot = r_del[i] + rsub_tot
        rtot <= 0 && continue
        if rand() < 1 - exp(-dt * rtot)
            u = rand() * rtot
            if u < r_del[i]
                to_delete[i] = true
            else
                u -= r_del[i]
                acc = 0.0
                chosen = 0
                for tok in 1:p.k
                    acc += r_sub[tok, i]
                    if u <= acc
                        chosen = tok
                        break
                    end
                end
                chosen == 0 && (chosen = x[i])
                sub_to[i] = chosen
            end
        end
    end

    ins_tok = zeros(Int, n + 1)
    for s in 0:n
        rtot = sum(@view r_ins[:, s + 1])
        rtot <= 0 && continue
        if rand() < 1 - exp(-dt * rtot)
            u = rand() * rtot
            acc = 0.0
            chosen = 0
            for tok in 1:p.k
                acc += r_ins[tok, s + 1]
                if u <= acc
                    chosen = tok
                    break
                end
            end
            chosen == 0 && (chosen = 1)
            ins_tok[s + 1] = chosen
        end
    end

    out = Int[]
    ins_tok[1] != 0 && push!(out, ins_tok[1])
    for i in 1:n
        if !to_delete[i]
            push!(out, sub_to[i] == 0 ? x[i] : sub_to[i])
        end
        ins_tok[i + 1] != 0 && push!(out, ins_tok[i + 1])
    end
    return DiscreteState(xt.K, out)
end


function validate_targets_vs_sampling(p::ModPIPProcess{T},
                                      xt::Vector{Int},
                                      x1::Vector{Int},
                                      t::T;
                                      N::Int = 50_000,
                                      seed::Int = 0) where {T}
    rng = MersenneTwister(seed)
    W = remaining_weights(p, t)

    acc_del = zeros(Float64, 1, length(xt))
    acc_sub = zeros(Float64, p.k, length(xt))
    acc_sub_weight = zeros(Float64, 1, length(xt))
    acc_ins_tok = zeros(Float64, p.k, length(xt) + 1)
    acc_ins_count = zeros(Float64, 1, length(xt) + 1)
    acc_ins_weight = zeros(Float64, 1, length(xt) + 1)

    for _ in 1:N
        events, _ = sample_alignment(rng, xt, x1, W)
        hard = hard_targets_from_events(p.k, x1, length(xt), events)
        acc_del .+= hard.del
        acc_sub .+= hard.sub
        acc_sub_weight .+= hard.sub_weight
        acc_ins_tok .+= hard.ins_tok
        acc_ins_count .+= hard.ins_count
        acc_ins_weight .+= hard.ins_weight
    end

    hard_avg = (del = acc_del ./ N,
                sub = let tmp = acc_sub ./ max.(acc_sub_weight, 1e-12)
                    for i in 1:size(tmp, 2)
                        if acc_sub_weight[1, i] == 0
                            fill!(view(tmp, :, i), 0.0)
                        end
                    end
                    tmp
                end,
                sub_weight = acc_sub_weight ./ N,
                ins_tok = let tmp = acc_ins_tok ./ max.(acc_ins_count, 1e-12)
                    for s in 1:size(tmp, 2)
                        if acc_ins_count[1, s] == 0
                            fill!(view(tmp, :, s), 0.0)
                        end
                    end
                    tmp
                end,
                ins_count = acc_ins_count ./ N,
                ins_weight = acc_ins_weight ./ N)

    soft = exact_targets(p, xt, x1, t)

    return (
        del_max = maximum(abs.(hard_avg.del .- soft.del)),
        sub_max = maximum(abs.(hard_avg.sub .- soft.sub)),
        subw_max = maximum(abs.(hard_avg.sub_weight .- soft.sub_weight)),
        ins_max = maximum(abs.(hard_avg.ins_tok .- soft.ins_tok)),
        insc_max = maximum(abs.(hard_avg.ins_count .- soft.ins_count)),
        hard = hard_avg,
        soft = soft
    )
end


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

function validate_bridge_step_consistency(p::ModPIPProcess,
                                          x0::DiscreteState,
                                          x1::DiscreteState;
                                          times = [0.2, 0.5, 0.8],
                                          Nbridge::Int = 5_000,
                                          Nstep::Int = 5_000,
                                          dt::Float64 = 0.01,
                                          seed::Int = 0)
    rng = MersenneTwister(seed)

    bridge_counts = Dict(t => Dict{Vector{Int}, Int}() for t in times)
    for t in times
        for _ in 1:Nbridge
            xt, _ = bridge_with_targets(rng, p, x0, x1, t)
            key = copy(tensor(xt))
            get!(bridge_counts[t], key, 0)
            bridge_counts[t][key] += 1
        end
    end

    step_counts = Dict(t => Dict{Vector{Int}, Int}() for t in times)
    grid = collect(0.0:dt:1.0)
    for _ in 1:Nstep
        xt = x0
        for (s1, s2) in zip(grid, grid[2:end])
            guide = Guide(p, s1, xt, x1)
            xt = step(p, xt, guide, s1, s2)
            for t in times
                if abs(s2 - t) < 1e-8
                    key = copy(tensor(xt))
                    get!(step_counts[t], key, 0)
                    step_counts[t][key] += 1
                end
            end
        end
    end

    return Dict(t => tvd(bridge_counts[t], Nbridge, step_counts[t], Nstep) for t in times)
end

end
