struct EditFlow <: DiscreteProcess
    k::Int                 # alphabet size (tokens 1..k)
    transform::Function    # maps unconstrained logits to positive rates
    κ::Function            # scheduler on [0,1] → [0,1] for bridge keep-probability
    dκ::Function           # derivative of κ
    padding_token::Int     # padding token id used in X1
    latent_token::Int      # latent placeholder used in Z0 / Zt
    bos_token::Int         # beginning-of-sequence token id (optional, unused by default)
end

EditFlow(k; transform = NNlib.softplus,
            κ = identity,
            dκ = t -> one(eltype(t)),
            padding_token::Int = k + 1,
            latent_token::Int = k + 2,
            bos_token::Int = 0) =
    EditFlow(k, transform, κ, dκ, padding_token, latent_token, bos_token)

"""
    EditFlow_cubic(k; transform=NNlib.softplus, padding_token=k+1, latent_token=k+2, bos_token=0)

Convenience constructor that uses a cubic scheduler:
- κ(t)  = t^3
- dκ(t) = 3t^2

Compatible with both scalar and broadcasted usage (the codebase typically calls `P.κ.(ts)` and also `P.κ(ts[j])`).
"""
EditFlow_cubic(k; transform = NNlib.softplus,
                  padding_token::Int = k + 1,
                  latent_token::Int = k + 2,
                  bos_token::Int = 0) = begin
    κ_cubic(t)  = t^3
    dκ_cubic(t) = 3 * (t^2)
    EditFlow(k, transform, κ_cubic, dκ_cubic, padding_token, latent_token, bos_token)
end


# Random padding of latent tokens to align the lengths of the sequences pairs (z0, z1)
# And afterwards batch the sequences with padding tokens into a single matrix (Z0, Z1)
function align_and_batch(P::EditFlow,
                                     x0s::Vector{<:DiscreteState},
                                     x1s::Vector{<:DiscreteState};
                                     rng=Random.default_rng())
    @assert length(x0s) == length(x1s)
    B = length(x0s)
    ltok = P.latent_token
    pad  = P.padding_token

    z0s = Vector{Vector{Int}}(undef, B)
    z1s = Vector{Vector{Int}}(undef, B)

    @inbounds for b in 1:B
        v0 = collect(tensor(x0s[b]))
        v1 = collect(tensor(x1s[b]))
        if length(v0) < length(v1)
            z0 = Vector{Int}(v0)
            d = length(v1) - length(v0)
            for _ in 1:d
                pos = rand(rng, 0:length(z0))
                insert!(z0, pos + 1, ltok)
            end
            z1 = Vector{Int}(v1)
        elseif length(v1) < length(v0)
            z1 = Vector{Int}(v1)
            d = length(v0) - length(v1)
            for _ in 1:d
                pos = rand(rng, 0:length(z1))
                insert!(z1, pos + 1, ltok)
            end
            z0 = Vector{Int}(v0)
        else
            z0 = Vector{Int}(v0)
            z1 = Vector{Int}(v1)
        end
        z0s[b] = z0
        z1s[b] = z1
    end

    maxlen = maximum(length.(z0s))
    Z0 = fill(Int(pad), maxlen, B)
    Z1 = fill(Int(pad), maxlen, B)
    @inbounds for b in 1:B
        lb = length(z0s[b])
        Z0[1:lb, b] .= z0s[b]
        Z1[1:lb, b] .= z1s[b]
    end

    return Z0, Z1
end

function interpolate_Z_elementwise(P::EditFlow,
                                  Z0::AbstractMatrix{<:Integer},
                                  Z1::AbstractMatrix{<:Integer},
                                  ts::AbstractVector)
    @assert size(Z0) == size(Z1)
    L, B = size(Z1)
    @assert length(ts) == B
    pad = P.padding_token

    Zt = similar(Z1)
    @inbounds for j in 1:B
        keep = clamp(P.κ(ts[j]), 0f0, 1f0)
        for i in 1:L
            z1 = Z1[i, j]
            if z1 == pad
                Zt[i, j] = pad                
            else
                Zt[i, j] = (rand(Float32) < keep) ? z1 : Z0[i, j]
            end
        end
    end

    ltok = P.latent_token
    filtered_cols = [filter(x -> !(x in (ltok, pad)), col) for col in eachcol(Zt)]
    Xt= hcat(map(col -> vcat(col, fill(pad, L - length(col))), filtered_cols)...)

    return Zt, Xt
end

function transition_mask_from_Xt(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    tokens = P.k
    pad = P.padding_token
    xt_len, B = size(Xt)
    T = ones(Float32, 2*tokens + 1, xt_len, B)
    for c in 1:B
        for i in 1:xt_len
            x = Xt[i, c]
            @assert x != P.latent_token
            if x == pad
                T[:, i, c] .= 0
            elseif x == P.bos_token
                @assert i == 1 #should only be BOS at position 1
                T[tokens+1:2*tokens+1, i, c] .= 0
            else
                # forbid sub-to-current-token only for valid tokens 1..K
                if 1 <= x <= tokens
                    T[tokens + x, i, c] = 0
                end
            end
        end
    end
    return T
end

# Drop-in replacement (renames ok): now returns (2K+1, L+1, B)
function transition_mask_from_Xt_gapwise(P::EditFlow, Xt::Matrix{Int})
    K   = P.k
    PAD = P.padding_token
    BOS = P.bos_token

    L, B = size(Xt)
    M = ones(Float32, 2K+1, L+1, B)

    for b in 1:B
        x = Xt[:,b]
        # real length (incl. BOS), sites 1..L_b; gaps 1..L_b+1
        Lb = count(t -> t != PAD, x)

        # 1) mask padding columns for sites/gaps beyond sequence
        #    - inserts valid only on gaps 1..Lb+1
        if Lb < L
            M[1:K, (Lb+2):(L+1), b] .= 0f0     # extra gaps off
            M[(K+1):(2K+1), (Lb+1):(L+1), b] .= 0f0  # sub/del beyond Lb off
        else
            # only the last pad gap (L+1) exists; keep it on for inserts, off for sub/del below
            nothing
        end
        # sub/del never valid at the last gap column
        M[(K+1):(2K+1), L+1, b] .= 0f0

        # 2) per-site constraints
        for i in 1:Lb
            t = x[i]
            # no self-sub, and typically no subs at BOS (policy)
            if t == BOS
                M[(K+1):(2K),  i, b] .= 0f0   # block all subs at BOS
                M[(2K+1),      i, b] = 0f0    # block delete BOS
            else
                M[K + t, i, b] = 0f0          # block self-sub to current t
            end
        end

        # 3) no ops on padding sites
        if Lb < L
            M[:, (Lb+1):L, b] .= 0f0
        end

        # 4) (policy) no pre-BOS insert (gap 1)
        M[1:K, 1, b] .= 0f0
    end
    return M
end

# Compute the remaining edits for the EditFlow as a Matrix
function remaining_edits(P::EditFlow, Zt::Matrix{Int}, Z1::Matrix{Int}, Xt::Matrix{Int}, dense=false)
    padding_token = P.padding_token
    latent_token = P.latent_token
    tokens = P.k
    (_, batch_size) = size(Z1)

    #filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    #batch_length = maximum(length, filtered_cols) 
    batch_length = size(Xt, 1)
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
    
    #Insert 
    #inserts = Z1.*(Zt .== latent_token)
    inserts = Z1 .* Int64.((Zt .== latent_token) .& (1 .≤ Z1 .≤ tokens))
    insert_edits = zeros(Float32, (tokens, batch_length, batch_size))
    insert_indices = findall(!iszero, inserts)
    insert_cols_to_update = pos[insert_indices]
    insert_rows_to_update = inserts[insert_indices]
    insert_samples_to_update = [idx[2] for idx in insert_indices]
    for i in 1:length(insert_rows_to_update)
        insert_edits[insert_rows_to_update[i], insert_cols_to_update[i], insert_samples_to_update[i]] += 1
    end
    dense_inserts = (insert_rows_to_update, insert_cols_to_update, insert_samples_to_update)

    #Substitution
    a = Zt .!= latent_token
    b = Z1 .!= latent_token
    c = Z1 .!= Zt
    subs = Z1.*(a .& b .& c)
    sub_edits = zeros(Float32, (tokens, batch_length, batch_size))
    sub_indices = findall(!iszero, subs)
    sub_cols_to_update = pos[sub_indices]
    sub_rows_to_update = subs[sub_indices]
    sub_samples_to_update = [idx[2] for idx in sub_indices]
    for i in 1:length(sub_rows_to_update)
        sub_edits[sub_rows_to_update[i], sub_cols_to_update[i], sub_samples_to_update[i]] = 1
    end
    dense_subs = (sub_rows_to_update .+ tokens, sub_cols_to_update, sub_samples_to_update)

    #Del
    dels = Z1 .== latent_token
    del_edits = zeros(Float32, (1, batch_length, batch_size))
    del_indices = findall(!iszero, dels)
    del_cols_to_update = pos[del_indices]
    del_samples_to_update = [idx[2] for idx in del_indices]
    for i in 1:length(del_cols_to_update)
        del_edits[1, del_cols_to_update[i], del_samples_to_update[i]] = 1
    end
    dense_dels = ((2*tokens+1).*ones(Int64, length(del_cols_to_update)), del_cols_to_update, del_samples_to_update) 

    if dense == true
        return (dense_inserts, dense_subs, dense_dels)
    else
        return vcat(insert_edits, sub_edits, del_edits)
    end

end

# Drop-in replacement (renames ok): now returns (2K+1, L+1, B)
function remaining_edits_gapwise(P::EditFlow, Zt::Matrix{Int}, Z1::Matrix{Int}, Xt::Matrix{Int})
    K  = P.k
    PAD = P.padding_token
    LAT = P.latent_token

    L, B = size(Xt)
    ins = zeros(Float32, K,   L+1, B) # gaps
    sub = zeros(Float32, K,   L,   B) # sites
    del = zeros(Float32, 1,   L,   B) # sites

    for b in 1:B
        Ztb = Zt[:,b]; Z1b = Z1[:,b]
        # real (site) tokens in Zt (BOS counts as real; PAD/LAT do not)
        real = (Ztb .!= LAT) .& (Ztb .!= PAD)
        sites_before = cumsum(real)         # length(Ztb)
        for r in 1:length(Ztb)
            zt = Ztb[r]; z1 = Z1b[r]
            # insert: LAT -> real
            if zt == LAT && (z1 != LAT && z1 != PAD)
                g = sites_before[r] + 1     # gap index in 1..L+1
                if 1 <= g <= L+1
                    ins[z1, g, b] += 1
                end
            # delete: real -> LAT
            elseif (zt != LAT && zt != PAD) && (z1 == LAT)
                s = sites_before[r]         # site index in 1..L
                if 1 <= s <= L
                    del[1, s, b] += 1
                end
            # substitute: real -> real, different
            elseif (zt != LAT && zt != PAD) && (z1 != LAT && z1 != PAD) && (zt != z1)
                s = sites_before[r]
                if 1 <= s <= L
                    sub[z1, s, b] += 1
                end
            end
        end
    end

    # pad sub/del with a zero column at gap L+1 so we can vcat on channel axis
    sub_pad = hcat(sub, zeros(Float32, size(sub,1), 1, size(sub,3)))
    del_pad = hcat(del, zeros(Float32, size(del,1), 1, size(del,3)))
    return vcat(ins, sub_pad, del_pad)      # (2K+1, L+1, B)
end


@inline function pick_index(w::AbstractVector{<:Real})::Int
    # treat negatives as zero; assert we have some mass
    cs = cumsum(max.(w, zero(eltype(w))))
    s  = cs[end]
    @assert isfinite(s) && s > 0 "pick_index: all weights ≤ 0 or non-finite"
    u = rand() * s
    return searchsortedfirst(cs, u)  # 1..length(w)
end

function step(P::EditFlow,
              Xt::DiscreteState{<:AbstractArray{<:Signed}},
              hat,
              s1::Real, s2::Real)

    @assert ndims(Xt.state) == 1 "EditFlow.step only supports 1D DiscreteState"

    # Rates
    pins, psub, pdel = part_output(P, P.transform(hat))   # (K,n+1,B), (K,n,B), (1,n,B)
    ins = Array(pins[:, :, 1])                            # (K, n+1) or (K, n)
    sub = Array(psub[:, :, 1]) 
    del = vec(Array(pdel[1, :, 1]))                       # (n,)  <-- fixed

    K, n = size(sub, 1), size(sub, 2)
    @assert size(ins, 1) == K
    @assert length(del) == n

    # Ensure gaps shape (K, n+1)
    ins_gaps = if size(ins, 2) == n + 1
        ins
    elseif size(ins, 2) == n
        tmp = similar(ins, K, n + 1)
        @inbounds for s in 0:n
            pos = clamp(s, 1, n)
            @views tmp[:, s + 1] .= ins[:, pos]
        end
        tmp
    else
        error("EditFlow.step: bad ins size $(size(ins))")
    end

    dt = float(s2 - s1)
    x = collect(tensor(Xt))  # Vector{Int}

    # Forbid self-substitutions
    if n > 0
        current_mask = zeros(eltype(sub), size(sub))
        @inbounds for i in 1:n
            tok = x[i]
            if 1 ≤ tok ≤ K
                current_mask[tok, i] = 1
            end
        end
        sub .*= (1 .- current_mask)
    end

    # Optionally forbid editing BOS explicitly
    if n > 0 && x[1] == P.bos_token
        sub[:, 1] .= 0
        del[1] = 0
    end

    # ---- site events (delete/sub) ----
    to_delete = falses(n)
    sub_to    = zeros(Int, n)
    @inbounds for i in 1:n
        r_del = del[i]
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0 && rand() < (1 - exp(-dt * r_tot))
            u = rand() * r_tot
            if u < r_del
                to_delete[i] = true
            elseif r_sub_total > 0
                sub_to[i] = pick_index(@view sub[:, i])
            end
        end
    end

    # ---- gap insertions (≤1 per gap) ----
    ins_tok = fill(0, n + 1)
    start_gap = (n > 0 && x[1] == P.bos_token) ? 1 : 0
    @inbounds for s in start_gap:n
        r_ins_total = sum(@view ins_gaps[:, s + 1])
        #println("r_ins_total", ins_gaps[:, s + 1])
        if r_ins_total > 0 && rand() < (1 - exp(-dt * r_ins_total))
            ins_tok[s + 1] = pick_index(@view ins_gaps[:, s + 1])
        end
    end

    # ---- build new sequence ----
    result = Int[]
    if ins_tok[1] != 0; push!(result, ins_tok[1]); end
    @inbounds for i in 1:n
        if !to_delete[i]
            a = (sub_to[i] == 0) ? x[i] : sub_to[i]
            push!(result, a)
        end
        if ins_tok[i + 1] != 0
            push!(result, ins_tok[i + 1])
        end
    end
    return DiscreteState(Xt.K, result)
end

function part_output(P::EditFlow, M::AbstractArray)
    K = P.k
    ins = M[1:K,:,:]
    sub = M[K+1:2K,:,:]
    del = M[2K+1:2K+1,:,:]
    return ins, sub, del
end


#=
function edit_loss(P::EditFlow,
                   M::AbstractArray,
                   transition_mask::AbstractArray,
                   edit_multiplier::AbstractArray,
                   scheduler_scaling;
                   op_mask=nothing,
                   eps=1e-8)
    R = P.transform(M)
    OM = isnothing(op_mask) ? one(eltype(R)) .* ones(eltype(R), size(R)) : op_mask
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))
    scl = reshape(scheduler_scaling, 1, 1, :)
    term2 = sum(scl .* edit_multiplier .* log.(R .+ eps); dims=(1,2))
    return mean(term1 .- term2)
end
=#
"""
    edit_loss(P::EditFlow, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=1e-8)

Loss matching the reference: mean(sum(transition_mask .* (op_mask .* R)) - sum(scheduler_scaling .* edit_multiplier .* log R)),
where R = transform(M).
Shapes:
- M, transition_mask, edit_multiplier, op_mask: (2K+1, n, B)
- scheduler_scaling: (1, B) or (B,) broadcastable to (1,1,B)
"""
function edit_loss(P::EditFlow,
                   M, transition_mask, edit_multiplier, scheduler_scaling;
                   op_mask=nothing, eps=1e-8)

    R = P.transform(M)                              # must be >= 0
    # (A) Optional op mask to apply symmetrically
    OM = isnothing(op_mask) ? one(eltype(R)) : op_mask

    # (B) Sum of valid outgoing rates
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))

    # (C) Logs only of positive rates (avoid NaN/Inf)
    R_logsafe = max.(R, eltype(R)(eps))             # clamp BEFORE log
    logR = log.(R_logsafe)

    scl = reshape(scheduler_scaling, 1, 1, :)       # (1,1,B)
    term2 = sum(scl .* (edit_multiplier .* OM) .* logR; dims=(1,2))

    return mean(term1 .- term2)
end


function edit_loss_gapwise(P::EditFlow,
    M, transition_mask, edit_multiplier, scheduler_scaling;
    op_mask=nothing, eps=1f-8)

    # 1) transform -> positiva satser
    R = P.transform(M)

    # 2) kapa hastigheter för att undvika Inf/NaN i både term1 och log-delen
    #    (justera RMAX vid behov, 1e3–1e4 funkar ofta bra)
    RMAX = 1f3
    R = clamp.(R, eps, RMAX)

    # 3) valfri op-mask
    OM = isnothing(op_mask) ? one(eltype(R)) : op_mask
    mask = transition_mask .* OM

    # 4) regularizer (sum av giltiga satser)
    #    -> per-batch summering: 1×1×B
    term1 = sum(mask .* R; dims=(1,2))

    # 5) data-del (log) – log säkert, redan kapat ovan
    logR = log.(R)
    scl  = reshape(scheduler_scaling, 1, 1, :)
    term2 = sum(scl .* (edit_multiplier .* OM) .* logR; dims=(1,2))

    # 6) normalisera med "antal aktiva positioner" för stabil skala
    active = sum(mask; dims=(1,2))
    loss_per_batch = (term1 .- term2) ./ (active .+ 1f-8)

    return mean(loss_per_batch)
end


"""
getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})

Build lmask of shape (xt_length, B) from padded Xt.
"""
function getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    padding_token = P.padding_token
    return Xt .!= padding_token
end

# Utility: pick_index over nonnegative weights (1D)
@inline function _pick_index1d!(rng::AbstractRNG, cs::AbstractVector{<:Real})::Int
    s = cs[end]
    @assert isfinite(s) && s > 0 "pick_index: total mass ≤ 0 or non-finite"
    u = rand(rng) * s
    return searchsortedfirst(cs, u)
end

# One CTMC Euler step over a single DiscreteState, using gap-wise insertions.
function step_gapwise(
    P::EditFlow,
    model,                 # your EditFlowModel (returns (2K+1, L+1, 1) logits)
    Xt::DiscreteState{<:AbstractVector{<:Integer}},
    t1::Real, t2::Real;    # times, with dt = t2 - t1 > 0
    rng::AbstractRNG = Random.default_rng(),
    transform::Function = P.transform,
)
    @assert t2 > t1
    x = collect(tensor(Xt))                # Vector{Int}
    K = P.k
    n = length(x)

    # Build masked state for the model
    lmask = trues(n)                       # all real positions valid
    cmask = trues(n)
    Xt_ms = MaskedState(DiscreteState(K, reshape(x, :, 1)), reshape(cmask, :, 1), reshape(lmask, :, 1))

    # Model forward (gap-wise head)
    M = model([Float32(t1)], Xt_ms)        # (2K+1, n+1, 1) logits
    R = transform(M)                       # positive rates
    ins = Array(R[1:K, :, 1])              # (K, n+1)
    sub = Array(R[K+1:2K, 1:n, 1])         # (K, n)
    del = vec(Array(R[2K+1, 1:n, 1]))      # (n,)

    # Enforce CTMC constraints: no self-sub, protect BOS
    if n > 0 && x[1] == P.bos_token
        sub[:, 1] .= 0
        del[1] = 0
        # also disallow pre-BOS insert (gap 1)
        ins[:, 1] .= 0
    end
    # forbid self-substitutions
    for i in 1:n
        tok = x[i]
        if 1 <= tok <= K
            sub[tok, i] = 0
        else
            sub[:, i] .= 0     # non-vocab token → block subs at site i
        end
    end

    dt = Float64(t2 - t1)

    # -------- sample site events (delete or substitute) --------
    to_delete = falses(n)
    sub_to    = zeros(Int, n)
    for i in 1:n
        r_del = max(del[i], 0.0)
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0 && rand(rng) < (1 - exp(-dt * r_tot))
            u = rand(rng) * r_tot
            if u < r_del
                to_delete[i] = true
            elseif r_sub_total > 0
                # draw new token from sub[:, i]
                cs = cumsum(@view sub[:, i])
                sub_to[i] = _pick_index1d!(rng, cs)
            end
        end
    end

    # -------- sample gap insertions (≤1 per gap) --------
    ins_tok = fill(0, n + 1)
    for g in 1:(n + 1)
        r_ins_total = sum(@view ins[:, g])
        if r_ins_total > 0 && rand(rng) < (1 - exp(-dt * r_ins_total))
            cs = cumsum(@view ins[:, g])
            ins_tok[g] = _pick_index1d!(rng, cs)
        end
    end

    # -------- build new sequence --------
    result = Int[]
    if ins_tok[1] != 0; push!(result, ins_tok[1]); end
    for i in 1:n
        if !to_delete[i]
            a = (sub_to[i] == 0) ? x[i] : sub_to[i]
            push!(result, a)
        end
        if ins_tok[i + 1] != 0
            push!(result, ins_tok[i + 1])
        end
    end

    return DiscreteState(Xt.K, result)
end

# Convenience: multi-step simulate from t=0→1 with a schedule of times `ts` (sorted)
function rollout_gapwise(P::EditFlow, model, x0::DiscreteState, ts::AbstractVector; rng=Random.default_rng())
    @assert issorted(ts) && first(ts) >= 0 && last(ts) <= 1
    x = x0
    for k in 1:length(ts)-1
        x = step_gapwise(P, model, x, ts[k], ts[k+1]; rng=rng)
    end
    return x
end
