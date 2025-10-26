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

"""
getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})

Build lmask of shape (xt_length, B) from padded Xt.
"""
function getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    padding_token = P.padding_token
    return Xt .!= padding_token
end