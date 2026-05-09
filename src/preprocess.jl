# pre-process data and initialize DATA structure
function init_problem_data(Dij_orig, D_orig, P_orig, torsions_orig)

    # perform basics tests on data
    check_param(size(D_orig) == size(Dij_orig), "Matrices D and Dij must have the same size")
    check_param(size(P_orig) == (maximum(Dij_orig),4), "P has an incorrect size.")
    check_param((minimum(P_orig[4:end,1:3]) > 0) && (maximum(abs.(P_orig[4:end,4])) <= 1), "P has incorrect entries")

    for i in 1:size(P_orig,1)
        check(maximum(P_orig[i,1:3]) < i, "P has incorrect entries")
    end

    check(size(Dij_orig,2) == 2, "Dij must be a #distances x 2 matrix with columns i, j")
    check(size(D_orig,2) == 2, "D must be a #distances x 2 matrix with columns lower_dij, upper_dij")

    Dij = deepcopy(Dij_orig)
    D = deepcopy(D_orig)

    # adjust Dij so that Dij[:,1] .< Dij[:,2] and verify if bounds are valid
    @inbounds @views for i = 1:size(Dij,1)
        if Dij[i,1] > Dij[i,2]
            Dij[i,1:2] .= Dij[i,2:-1:1]
        end
        check(D[i,1] <= D[i,2], "Invalid lower/upper bounds on distance between vertices $(Dij[i,1]) and $(Dij[i,2])")
    end

    # consolidate repeated distances
    idx = 1
    @inbounds @views while (idx <= size(Dij,1))
        # rows (> idx) that represent the same distance as row idx
        same = ((Dij[1:end,1] .== Dij[idx,1]) .& (Dij[1:end,2] .== Dij[idx,2]))
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

    check(any(D[:,2] .> D[:,1]), "Instance is exact, enumerative strategies are recommended")

    # number of atoms and distances after pre-processing
    nv = maximum(Dij)
    nd = size(Dij,1)

    # P has 4 columns: predecessors (cols 1 to 3) and branching signs (col 4)
    P = Matrix{Int64}(undef, nv, 4)
    @views P[1:end,1:3] .= P_orig[1:end,1:3]
    @views if size(P,2) >= 4
        # the signs are provided
        P[1:end,4] .= P_orig[1:end,4]
    else
        P[1:end,4] .= 0
    end

    # exact distances between atoms and their immediate predecessors
    exactds = Float64[]
    # atom 1
    push!(exactds, 0.0)
    for k in 2:nv
        d = D[(Dij[:,1] .== P[k,1]) .& (Dij[:,2] .== k),1]
        check(!isempty(d), "Distance from $(P[k,1]) and $(k) missing")
        push!(exactds, d[1])
    end

    # construct the map (i,j) to the position in D
    ij_to_D = spzeros(Int64, nv, nv)
    @inbounds @views for k in 1:nd
        ij_to_D[ Dij[k,1], Dij[k,2] ] = k
    end
    ij_to_D = Symmetric(ij_to_D, :U)

    return DATA(
        nv,
        nd,
        Dij,
        D,
        P,
        torsions_orig,
        exactds,
        ij_to_D
    )
end

# check if all necessary distances are provided
function check_necessarydistances(data::DATA, par::MDGP_PARAMETERS)
    exact = (abs.(data.D[:,2] - data.D[:,1]) .<= par.tol_exact)

    # exact distances ( d(i1,i) and d(i2,i) )
    @inbounds @views for i = 3:data.nv
        i1 = data.ij_to_D[i,data.P[i,1]]
        i2 = data.ij_to_D[i,data.P[i,2]]
        check((i1 > 0) && exact[i1], "Necessary exact distances from vertex $(i) were not provided")
        check((i2 > 0) && exact[i2], "Necessary exact distances from vertex $(i) were not provided")
    end

    # distances d(i3,i) (exact or interval)
    @inbounds @views for i = 4:data.nv
        check(data.ij_to_D[i,data.P[i,3]] > 0, "Exact or interval distance between $(data.P[i,3]) and $(i) was not provided")
    end
end

# tight bounds, returning true if an infeasibility was identified
function tight_bounds!(data::DATA, par::MDGP_PARAMETERS, verbose)
    modified = true

    while (modified)
        modified = false
        @inbounds @views for i = 4:data.nv
            i3 = data.P[i,3]
            k = data.ij_to_D[i3,i]

            if data.D[k,2] - data.D[k,1] > par.tol_exact
                i1 = data.P[i,1]
                i2 = data.P[i,2]

                d10 = d(i1, i,  data)[1]
                d20 = d(i2, i,  data)[1]
                d21 = d(i2, i1, data)[1]
                d31 = d(i3, i1, data)[1]
                d32 = d(i3, i2, data)[1]

                d30_0 = d30(d10, d32, d31, d21, d20, -1.0)
                d30_1 = d30(d10, d32, d31, d21, d20,  1.0)

                d30L = min(d30_0,d30_1)
                d30U = max(d30_0,d30_1)

                # lower bound
                if data.D[k,1] < d30L
                    if verbose > 1
                        @printf("lower bound of d_%d,%d was improved from %.6lf to %.6lf\n",i3,i,data.D[k,1],d30L)
                    end
                    data.D[k,1] = d30L
                    modified = true
                end

                # upper bound
                if data.D[k,2] > d30U
                    if verbose > 1
                        @printf("upper bound of d_%d,%d was improved from %.6lf to %.6lf\n",i3,i,data.D[k,2],d30U)
                    end
                    data.D[k,2] = d30U
                    modified = true
                end

                if data.D[k,1] > data.D[k,2]
                    if verbose > 1
                        @warn "After tightening the bounds, the distance between $(i3) and $(i) was found to be inconsistent"
                    end
                    return true
                end

                if data.D[k,2] - data.D[k,1] <= par.tol_exact
                    data.D[k,1:2] .= (data.D[k,2] + data.D[k,1])/2
                end
            end
        end
    end
    return false
end
