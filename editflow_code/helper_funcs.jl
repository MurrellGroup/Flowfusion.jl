# Sampling helpers
function state_to_PM_string(P::FF.EditFlow, st::FF.DiscreteState)
    xs = FF.tensor(st)
    pad = hasfield(typeof(P), :padding_token) ? P.padding_token : -1
    io = IOBuffer()
    @inbounds for tok in xs
        if tok == 0
            write(io, '>')
        elseif tok == pad
            continue
        else
            write(io, AA20[tok])
        end
    end
    return String(take!(io))
end

states_to_PM_strings(P::FF.EditFlow, sts::Vector{<:FF.DiscreteState}) =
    [state_to_PM_string(P, s) for s in sts]

function sample_gen_n(P::FF.EditFlow, model; n::Int=10, ts=0f0:0.01f0:1f0, rng=Random.default_rng())
    gens = Vector{Any}(undef, n)
    for i in 1:n
        x0s, _, _ = make_minibatch(1, P; rng=rng)
        bos = P.bos_token
        toks = collect(FF.tensor(x0s[1]))
        x0_bos = FF.DiscreteState(P.k, vcat([bos], toks))
        gens[i] = FF.gen(P, x0_bos, model, ts)
    end
    return gens
end

function sample_gen_10_strings(P::FF.EditFlow, model; ts=0f0:0.01f0:1f0)
    outs = sample_gen_n(P, model; n=20, ts=ts)
    return map(outs) do o
        if o isa FF.DiscreteState
            state_to_PM_string(P, o)
        elseif o isa AbstractVector{<:FF.DiscreteState}
            states_to_PM_strings(P, o)
        else
            o
        end
    end
end

final_state_from_gen(res) = res isa FF.DiscreteState ? res : (res isa AbstractVector{<:FF.DiscreteState} ? last(res) : res)

function to_aa_tokens(P::FF.EditFlow, st::FF.DiscreteState)
    xs = FF.tensor(st)
    pad = hasfield(typeof(P), :padding_token) ? P.padding_token : -1
    toks = Vector{Int}()
    @inbounds for tok in xs
        if tok == 0 || tok == pad
            continue
        else
            push!(toks, Int(tok))
        end
    end
    return toks
end