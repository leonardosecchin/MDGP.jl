# Z = X
@inline function cp!(
    idxs,
    Z::AbstractVecOrMat{Float64},
    X::AbstractVecOrMat{Float64}
)
    @inbounds for i in idxs
        Z[i] = X[i]
    end
end
@inline cp!(
    Z::AbstractVecOrMat{Float64}, X::AbstractVecOrMat{Float64}
) = cp!(1:length(Z), Z, X)

# Z = a*X
@inline function aX!(
    idxs,
    Z::AbstractVecOrMat{Float64},
    a::Float64,
    X::AbstractVecOrMat{Float64}
)
    @inbounds for i in idxs
        Z[i] = a*X[i]
    end
end
@inline aX!(
    Z::AbstractVecOrMat{Float64},
    a::Float64,
    X::AbstractVecOrMat{Float64}
) = aX!(1:length(Z), Z, a, X)

# Z = X + Y
@inline function XpY!(
    idxs,
    Z::AbstractVecOrMat{Float64},
    X::AbstractVecOrMat{Float64},
    Y::AbstractVecOrMat{Float64}
)
    @inbounds for i in idxs
        Z[i] = X[i] + Y[i]
    end
end
@inline XpY!(
    Z::AbstractVecOrMat{Float64},
    X::AbstractVecOrMat{Float64},
    Y::AbstractVecOrMat{Float64}
) = XpY!(1:length(Z), Z, X, Y)

# Z = X + a*Y
@inline function XpaY!(
    idxs,
    Z::AbstractVecOrMat{Float64},
    X::AbstractVecOrMat{Float64},
    a::Float64,
    Y::AbstractVecOrMat{Float64}
)
    @inbounds for i in idxs
        Z[i] = X[i] + a*Y[i]
    end
end
@inline XpaY!(
    Z::AbstractVecOrMat{Float64},
    X::AbstractVecOrMat{Float64},
    a::Float64,
    Y::AbstractVecOrMat{Float64}
) = XpaY!(1:length(Z), Z, X, a, Y)

# Z = dot(X,Y)
@inline function XdotY(
    idxs,
    X::AbstractVecOrMat{Float64},
    Y::AbstractVecOrMat{Float64}
)
    d = 0.0
    @inbounds for i in idxs
        d += X[i] * Y[i]
    end
    return d
end
@inline XdotY(
    X::AbstractVecOrMat{Float64},
    Y::AbstractVecOrMat{Float64}
) = XdotY(1:length(X), X, Y)

# convert v into a UnitRange if possible
function consec_range(v)
    if isempty(v)
        return v
    else
        sort!(v)
        return ifelse(v[end] - v[1] + 1 == length(v), v[1]:v[end], v)
    end
end
