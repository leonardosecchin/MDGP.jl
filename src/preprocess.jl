function init_P(P_orig, nv)
    P = Matrix{Int64}(undef, nv, 4)
    @views P[1:end,1:3] .= P_orig[1:end,1:3]
    if size(P,2) >= 4
        # the signs are provided
        @views P[1:end,4] .= P_orig[1:end,4]
    else
        P[1:end,4] .= 0
    end
    return P
end

# perform basics tests on data
function check_basics(Dij, D, P)
    @assert size(D) == size(Dij) "Matrices D and Dij must have the same size"
    @assert size(P) == (maximum(Dij),4) "P has an incorrect size."
    @assert (minimum(P[4:end,1:3]) > 0) && (maximum(abs.(P[4:end,4])) <= 1) "P has incorrect entries"

    for i in 1:size(P,1)
        @assert (maximum(P[i,1:3]) < i) "P has incorrect entries"
    end

    @assert size(Dij,2) == 2 "Dij must be a #distances x 2 matrix with columns i, j"
    @assert size(D,2) == 2 "D must be a #distances x 2 matrix with columns lower_dij, upper_dij"

    # adjust Dij so that Dij[:,1] .< Dij[:,2] and verify if bounds are valid
    @inbounds @views for i = 1:size(Dij,1)
        if Dij[i,1] > Dij[i,2]
            Dij[i,1:2] .= Dij[i,2:-1:1]
        end
        @assert (D[i,1] <= D[i,2]) "Invalid lower/upper bounds on distance between vertices $(Dij[i,1]) and $(Dij[i,2])"
    end
end

# check if all necessary distances are provided
function check_necessarydistances(nv, D, P, ij_to_D, tol_exact)
    exact = (abs.(D[:,2] - D[:,1]) .<= tol_exact)

    # exact distances ( d(i1,i) and d(i2,i) )
    @inbounds @views for i = 3:nv
        i1 = ij_to_D[i,P[i,1]]
        i2 = ij_to_D[i,P[i,2]]
        @assert (i1 > 0) && exact[i1] "Necessary exact distances from vertex $(i) were not provided"
        @assert (i2 > 0) && exact[i2] "Necessary exact distances from vertex $(i) were not provided"
    end

    # distances d(i3,i) (exact or interval)
    @inbounds @views for i = 4:nv
        @assert ij_to_D[i,P[i,3]] > 0 "Exact or interval distance between $(P[i,3]) and $(i) was not provided"
    end
end

# consolidate repeated distances
function consolidate_distances!(Dij, D)
    idx = 1
    @inbounds while (idx <= size(Dij,1))
        # rows (> idx) that represent the same distance as row idx
        @views same = ((Dij[1:end,1] .== Dij[idx,1]) .& (Dij[1:end,2] .== Dij[idx,2]))
        same[1:idx] .= 0

        # If two or more rows refer to a same distance, we take the tightest bound
        if any(same)
            # tightest bounds
            D[idx,1] = maximum(D[same,1])
            D[idx,2] = minimum(D[same,2])

            # remove repeated rows
            Dij = Dij[.!same,1:2]
            D   = D[.!same,1:2]
        end

        # pass to the next row
        idx += 1
    end
end

# tight bounds, returning true if an infeasibility was identified
function tight_bounds!(nv, D, P, ij_to_D, tol_exact, verbose)
    modified = true

    while (modified)
        modified = false
        @inbounds @views for i = 4:nv
            i3 = P[i,3]
            k = ij_to_D[i3,i]

            if D[k,2] - D[k,1] > tol_exact
                i1 = P[i,1]
                i2 = P[i,2]

                d10 = d(i1, i,  D, ij_to_D)[1]
                d20 = d(i2, i,  D, ij_to_D)[1]
                d21 = d(i2, i1, D, ij_to_D)[1]
                d31 = d(i3, i1, D, ij_to_D)[1]
                d32 = d(i3, i2, D, ij_to_D)[1]

                d30_0 = d30(d10, d32, d31, d21, d20, -1.0)
                d30_1 = d30(d10, d32, d31, d21, d20,  1.0)

                d30L = min(d30_0,d30_1)
                d30U = max(d30_0,d30_1)

                # lower bound
                if D[k,1] < d30L
                    if verbose > 1
                        @printf("lower bound of d_%d,%d was improved from %.6lf to %.6lf\n",i3,i,D[k,1],d30L)
                    end
                    D[k,1] = d30L
                    modified = true
                end

                # upper bound
                if D[k,2] > d30U
                    if verbose > 1
                        @printf("upper bound of d_%d,%d was improved from %.6lf to %.6lf\n",i3,i,D[k,2],d30U)
                    end
                    D[k,2] = d30U
                    modified = true
                end

                if D[k,1] > D[k,2]
                    if verbose > 0
                        @warn "After tightening the bounds, the distance between $(i3) and $(i) was found to be incosistent (lower bound = $(D[k,1]), upper bound = $(D[k,2]))"
                    end
                    return true
                end

                if D[k,2] - D[k,1] <= tol_exact
                    D[k,1:2] .= (D[k,2] + D[k,1])/2
                end
            end
        end
    end
    return false
end
