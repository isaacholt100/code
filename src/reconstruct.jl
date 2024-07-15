module reconstruct

chebyshevt(n::Integer, x::Real) = cos(n * acos(x))
chebbasisshifted(a::Real, b::Real) = n -> x -> chebyshevt(n, 2(x - a)/(b - a) - 1)
twochebbasis(n::Integer, m::Integer) = x -> chebyshevt(n, x[1]) * chebyshevt(m, x[2])

function espindices(max::Integer, length::Integer)
    @assert length >= 1
    @assert max >= length
    if length == 1
        return [[i] for i=1:max]
    end
    if max == length
        return [[i for i=1:max]]
    end
    a = espindices(max - 1, length)
    b = espindices(max - 1, length - 1)
    for idx in b
        push!(idx, max)
    end
    return vcat(a, b)
end

function elemsympoly(n::Integer, r::Integer)
    if r > n
        return x -> 0.0
    end
    if r == 0
        return x -> 1.0
    end
    function out(x::Vector{T})::T where T <: Real
        @assert length(x) == n
        return sum(prod(x[i] for i in idx) for idx in espindices(n, r))
    end
    return out
end

function estimatematchedcusppoint(closepoints::Vector{Tuple{T, Real}})::T where T
    estimate, _ = findmin(x -> x[1], closepoints)
    return estimate[0]
end

end # module reconstruct