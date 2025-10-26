function combine(
    elements::A, new_elements::A,
    counts::AbstractVector{Int}=[size(new_elements)[end]],
    positions::AbstractVector{Int}=[size(elements)[end]],
    deletions::AbstractVector{Bool}=fill(false, size(elements)[end]);
    offset::Int=0
) where A<:AbstractArray
    D = ndims(elements) - offset
    @assert length(deletions) == size(elements)[end]

    insertions = Dict{Int,Int}()
    for (p, n) in zip(positions, counts)
        insertions[p] = get(insertions, p, 0) + n
    end

    num_kept = count(!, deletions)
    num_new = sum(values(insertions))
    combined_len = num_kept + num_new

    other_dims = size(elements)[1:end-1]
    combined_size = (other_dims..., combined_len)
    combined_elements = similar(elements, combined_size)

    dst_idx = 1
    new_idx = 1

    if haskey(insertions, 0)
        n = insertions[0]
        if new_idx + n - 1 <= size(new_elements)[end] && dst_idx + n - 1 <= combined_len
            selectdim(combined_elements, D, dst_idx:dst_idx+n-1) .= selectdim(new_elements, D, new_idx:new_idx+n-1)
            dst_idx += n
            new_idx += n
        end
    end

    for src_idx in 1:size(elements)[end]
        if !deletions[src_idx] && dst_idx <= combined_len
            selectdim(combined_elements, D, dst_idx) .= selectdim(elements, D, src_idx)
            dst_idx += 1
        end

        if haskey(insertions, src_idx)
            n = insertions[src_idx]
            if new_idx + n - 1 <= size(new_elements)[end] && dst_idx + n - 1 <= combined_len
                selectdim(combined_elements, D, dst_idx:dst_idx+n-1) .= selectdim(new_elements, D, new_idx:new_idx+n-1)
                dst_idx += n
                new_idx += n
            end
        end
    end

    if dst_idx <= combined_len
        return selectdim(combined_elements, D, 1:dst_idx-1)
    else
        return combined_elements
    end
end

function combine(elements::DiscreteState, new_elements::DiscreteState, args...; kws...)
    @assert elements.K == new_elements.K "Number of categories must match"
    combined_state = combine(elements.state, new_elements.state, args...; kws...)
    return DiscreteState(elements.K, combined_state)
end

function combine(elements::ContinuousState, new_elements::ContinuousState, args...; kws...)
    combined_state = combine(elements.state, new_elements.state, args...; kws...)
    return ContinuousState(combined_state)
end

function combine(elements::ManifoldState, new_elements::ManifoldState, args...; kws...)
    @assert elements.M == new_elements.M "Manifolds must match"
    combined_state = combine(elements.state, new_elements.state, args...; kws...)
    return ManifoldState(elements.M, combined_state)
end

function combine(elements::MaskedState, new_elements::MaskedState, args...; kws...)
    combined_state = combine(elements.S, new_elements.S, args...; kws...)
    combined_cmask = combine(elements.cmask, new_elements.cmask, args...; kws...)
    combined_lmask = combine(elements.lmask, new_elements.lmask, args...; kws...)
    return MaskedState(combined_state, combined_cmask, combined_lmask)
end

function combine(elements::Tuple, new_elements::Tuple, args...; kws...)
    return map((e...) -> combine(e..., args...; kws...), elements, new_elements)
end

selectlastdim(x::AbstractArray, i; offset::Int = 0) = copy(selectdim(x, ndims(x) - offset, i))

selectlastdim(x::DiscreteState, i; kws...) = DiscreteState(x.K, selectlastdim(x.state, i; kws...))
selectlastdim(x::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}, i; kws...) = begin
    labels = onecold(x.state, 1:x.K)
    sel = selectlastdim(labels, i; kws...)
    DiscreteState(x.K, onehotbatch(sel, 1:x.K))
end
selectlastdim(x::ContinuousState, i; kws...) = ContinuousState(selectlastdim(x.state, i; kws...))
selectlastdim(x::ManifoldState, i; kws...) = ManifoldState(x.M, selectlastdim(x.state, i; kws...))
selectlastdim(x::MaskedState, i; kws...) = MaskedState(selectlastdim(x.S, i; kws...), selectlastdim(x.cmask, i; kws...), selectlastdim(x.lmask, i; kws...))
selectlastdim(x::Tuple, i; kws...) = map(v -> selectlastdim(v, i; kws...), x)

lastsize(x::DiscreteState; offset::Int = 0) = size(x.state)[end - offset]
lastsize(x::ContinuousState; offset::Int = 0) = size(x.state)[end - offset]
lastsize(x::ManifoldState; offset::Int = 0) = size(x.state)[end - offset]
lastsize(x::MaskedState; offset::Int = 0) = lastsize(x.S; offset)
lastsize(x::Tuple; offset::Int = 0) = only(unique(map(v -> lastsize(v; offset), x)))

export lastsize, selectlastdim
