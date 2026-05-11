# convert v into a UnitRange if possible
function consec_range(v)
    if isempty(v)
        return v
    else
        sort!(v)
        return ifelse(v[end] - v[1] + 1 == length(v), v[1]:v[end], v)
    end
end

# return [ lower d(i,j) ; upper d(i,j)] from D. Distance d(i,j) must exists in D
@inline function d(i, j, data::DATA)
    return @inbounds @views data.D[data.ij_to_D[i,j], 1:end]
end


# bond length  : r_i = d_{i-1,i}
# bond angle   : θ_i = "angle between segments i-2 --- i-1 and i-1 --- i"
# torsion angle: ω_i = "angle between the normals through the planes deﬁned
#                by the atoms i-3, i-2, i-1 and i-2, i-1, i"

# given vertices i-2, i-1, i, compute the cosine of the bond angle θ_i
# by the law of cosines:
#
# cos(θ_i) = d_{i-1,i}^2 + d_{i-2,i-1}^2 - d_{i-2,i}^2
#            -----------------------------------------
#                      2 d_{i-1,i} d_{i-2,i-1}

function cosθ(d10, d20, d21)
    cosine = (d10^2 + d21^2 - d20^2)/(2.0*d21*d10)
    return sign(cosine) * min(abs(cosine),1.0)
end


# Given vertices i-3, i-2, i-1, i, compute the cosine of the torsion angle:
#
# cos(ω_i) = r^2_{i-2} + d^2_{i-2,i} - 2 r_{i-2} d_{i-2,i} cos(θ_{i-1}) cos(θ_i) - d^2_{i-3,i}
#            ---------------------------------------------------------------------------------
#                                    2 r_{i-2} d_{i-2,i} sin(θ_{i-1}) sin(θ_i)
#
#              = cosθ(d32,d30,d20) - cos(θ_{i-1}) cos(θ_i)
#                -----------------------------------------
#                          sin(θ_{i-1}) sin(θ_i)
#
# The (possible) interval distance d30 = d^2_{i-3,i} must be fixed outside this
# function. See Lavor, Liberti, Maculan. A note on "A branch-and-prune algorithm
# for the molecular distance geometry problem". Intl. Trans. in Op. Res. 18
# (2011) 751-752

function cosω(d10, d32, d31, d21, d20, d30)
    cosθ1 = cosθ(d21,d31,d32)
    cosθ0 = cosθ(d20,d10,d21)
    r     = cosθ(d32,d30,d20)

    cosine = (r - cosθ1 * cosθ0) / (sqrt(1.0 - cosθ1^2) * sqrt(1.0 - cosθ0^2))
    return sign(cosine) * min(abs(cosine),1.0)
end


# Compute the distance d_{i-3,i} given all exact distances between vertices i-3, i-2, i-1, i and the cosine of the torsion angle ω_i

function d30(d10, d32, d31, d21, d20, cosω)
    cosθ1 = cosθ(d32,d31,d21)
    cosθ0 = cosθ(d20,d10,d21)

    sinθ0 = sqrt(1.0 - cosθ0^2)
    sinθ1 = sqrt(1.0 - cosθ1^2)

    # isolate d_{i-3,i} in the formula for the cosine of ω
    return sqrt(d32^2 + d20^2 - 2.0*d32*d20*(cosω*sinθ0*sinθ1 + cosθ0*cosθ1))
end


# Mean/Largest Distance Error (MDE/LDE)
function LDE(data::DATA, idxD, X::Matrix{Float64})
    lde = 0.0

    @inbounds @views for k in idxD
        dist = euclidean(X[1:3,data.Dij[k,1]], X[1:3,data.Dij[k,2]])
        L,U = data.D[k,1:2]
        if L == U
            lde = max(lde, abs(L - dist)/L)
        else
            lde = max(lde, max( max(L - dist, 0.0)/L, max(dist - U, 0.0)/U ) )
        end
    end

    return lde
end

function MDE_LDE(data::DATA, idxD, X::Matrix{Float64})
    mde = 0.0
    lde = 0.0

    @inbounds @views for k in idxD
        dist = euclidean(X[1:3,data.Dij[k,1]], X[1:3,data.Dij[k,2]])
        L,U = data.D[k,1:2]
        if L == U
            mde += abs(L - dist)/L
            lde = max(lde, abs(L - dist)/L)
        else
            mde += max(L - dist, 0.0)/L + max(dist - U, 0.0)/U
            lde = max(lde, max( max(L - dist, 0.0)/L, max(dist - U, 0.0)/U ) )
        end
    end

    return mde/length(idxD), lde
end

MDE_LDE(data::DATA, X::Matrix{Float64}) = MDE_LDE(data, 1:size(data.D,1), X)

function MDE_LDE(data::DATA, idxD, dists::Vector{Float64})
    mde = 0.0
    lde = 0.0

    @inbounds @views for k in idxD
        L,U = data.D[k,1:2]
        if L == U
            mde += abs(L - dists[k])/L
            lde = max(lde, abs(L - dists[k])/L)
        else
            mde += max(L - dists[k], 0.0)/L + max(dists[k] - U, 0.0)/U
            lde = max(lde, max( max(L - dists[k], 0.0)/L, max(dists[k] - U, 0.0)/U ) )
        end
    end

    return mde/length(idxD), lde
end


# centralize conformation w.r.t. all atoms
function centralize!(X, Xout, idx)
    @inbounds @views work = sum(X[1:3,idx], dims=2)./length(idx)
    @inbounds @views for i in idx
        @. Xout[1:3,i] = X[1:3,i] - work
    end
end


# RMSD between two conformations w.r.t. given atoms indices,
# conformations are considered centralized
function rmsd_protein(X1, X2, idx)
    @inbounds @views SVD = svd(X2[1:3,idx] * X1[1:3,idx]')
    return @inbounds @views (1.0/sqrt(length(idx))) * norm(SVD.U * SVD.Vt * X1[1:3,idx] - X2[1:3,idx], 2)
end
