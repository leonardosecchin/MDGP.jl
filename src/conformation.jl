# rotation matrix U w.r.t. the already positioned vertices l-3, l-2 and l-1
# U = [ xx  yy  zz ]
# with
# xx = v1/|v1|
# zz = v1 x v2/|v1 x v2|
# yy = zz x xx
# where
# v1 = X[:,l-1] .- X[:,l-2]
# v2 = X[:,l-3] .- X[:,l-2]
# see Gonçalves, Mucherino. Discretization orders and efficient computation of cartesian coordinates for distance geometry. Optim Lett (2014) 8:2111-2125
@inline function computeU!(U, X, l1, l2, l3)
    @inbounds @views begin
        @. U[1:3,1] = X[1:3,l1] - X[1:3,l2] #v1
        @. U[1:3,3] = X[1:3,l3] - X[1:3,l2] #v2
        U[1:3,3] .= cross(U[1:3,1], U[1:3,3]) #v1 x v2
        U[1:3,2] .= cross(U[1:3,3], U[1:3,1]) #(v1 x v2) x v1
        # normalize
        U[1:3,1] ./= norm(U[1:3,1])
        U[1:3,2] ./= norm(U[1:3,2])
        U[1:3,3] ./= norm(U[1:3,3])
    end
end

# construct the conformation according to 'fixed_torsions'
function construct_conformation!(
    l,
    data::DATA,
    X,
    fixed_torsions,
    adj,
    maxtrials,
    par::MDGP_PARAMETERS,
    fixsign,
    U
)

    @inbounds @views if (l <= size(X,2))

        # predecessors of l
        l1, l2, l3 = data.P[l,1:3]

        # distances
        d10 = d(l1, l, data)[1]   # exact
        d20 = d(l2, l, data)[1]   # exact
        d21 = d(l2, l1, X)        # l1 and l2 already positioned

        # cosine/sine of the bond angle θ_l
        cosθ_l = cosθ(d10, d20, d21)
        sinθ_l = sqrt(1.0 - cosθ_l^2)

        # rotation matrix U w.r.t. vertices l-3, l-2, l-1
        computeU!(U, X, l1, l2, l3)

        ntrials = 0
        besttorsion = fixed_torsions[l]
        bestpartial_lde = Inf

        while (true)
            if (ntrials >= maxtrials) && (ntrials > 0)
                break
            end

            # cosine/sine of the torsion angle ω_l
            ω_l = pi * fixed_torsions[l] / 180.0
            sinω_l, cosω_l = sincos(ω_l)

            X[1:3,l] .= X[1:3,l1] .+ d10*U*[-cosθ_l ;
                                             sinθ_l*cosω_l ;
                                             sinθ_l*sinω_l ]

            if maxtrials == 0
                # just construct the conformation with the given angles
                break
            end

            if (data.P[l,4] != 0) && (data.torsions[l,2] == 0)
                # there is only possible angle, skip
                break
            end

            # partial measure of quality
            partial_lde = compute_partial_lde(l, data, X, adj)

            # stop trials if LDE is small
            if partial_lde <= par.tol_lde^2
                break
            end

            if partial_lde < bestpartial_lde
                # a better configuration was found
                bestpartial_lde = partial_lde
                besttorsion = fixed_torsions[l]
            end

            if (data.P[l,4] == 0) && (data.torsions[l,2] == 0)
                # there is two possibilities, so we just choose the other,
                # which corresponds to flip the sign
                if fixsign
                    break
                end
                fixed_torsions[l] *= -1.0

                # indicates that no retry is necessary
                ntrials = maxtrials
            else
                # sort a new torsion angle
                if fixsign
                    sgn = Int64(sign(fixed_torsions[l]))
                else
                    sgn = (data.P[l,4] == 0) ? rand([-1,1]) : data.P[l,4]
                end
                fixed_torsions[l] = sort_torsion_angle(l, data, sgn)
            end

            ntrials += 1
        end

        # best angle found so far
        fixed_torsions[l] = besttorsion

        # pass to the next level
        construct_conformation!(
            l+1,
            data,
            X, fixed_torsions, adj,
            maxtrials, par, fixsign, U
        )
    else
        # end of construction
        return true
    end
end

# try to improve a given conformation w.r.t. "objective" by flipping signs of given atoms
function improve_conformation!(
    idxD,
    data::DATA,
    X,
    fixed_torsions,
    adj,
    par::MDGP_PARAMETERS,
    U
)

    best_lde = LDE(data, idxD, X)

    if best_lde <= par.tol_lde
        return best_lde
    end

    @inbounds for pass in 1:par.N_impr
        changed = false

        # vertex from which we have to recompute the conformation
        update_from_v = 4
        recompute = false

        for v in 4:length(fixed_torsions)
            # Test whether the sign of the torsion angle can be flipped.
            # This occurs when
            #   P[v,4] = 0  OR
            #   P[v,4] = 0  AND  minus "fixed torsion angle" lies in the torsion angle interval
            if (data.P[v,4] != 0) &&
               (
               (-fixed_torsions[v] < data.P[v,4]*data.torsions[v,1] - data.torsions[v,2]) ||
               (-fixed_torsions[v] > data.P[v,4]*data.torsions[v,1] + data.torsions[v,2])
               )
                continue
            end

            # flip sign
            fixed_torsions[v] *= -1.0

            # if the conformation is up to date, we only need to compute it from v
            if !recompute
                update_from_v = v
            end

            # reconstruct conformation from the last non udpated vertex
            construct_conformation!(
                update_from_v, data, X, fixed_torsions, adj, par.N_tors, par, true, U
            )

            new_lde = LDE(data, idxD, X)

            if new_lde < best_lde
                best_lde = new_lde
                if new_lde <= par.tol_lde
                    return new_lde
                end
                changed = true
                recompute = false
            else
                # back to the previous conformation and indicate that
                # the conformation should be recomputed from v in the next step
                fixed_torsions[v] *= -1.0
                update_from_v = v
                recompute = true
            end
        end

        # the conformation needs to be udpated; no torsion angle will be re-sorted
        if recompute
            construct_conformation!(
                update_from_v, data, X, fixed_torsions, adj, 0, par, true, U
            )
        end

        if !changed
            break
        end
    end
    return best_lde
end

function start_conformation(data::DATA)
    X = Matrix{Float64}(undef, 3, data.nv)

    # construct first three points
    d10 = d(2, 3, data)[1]    # d_{3-1,3}   = d_{2,3}
    d20 = d(1, 3, data)[1]    # d_{3-2,3}   = d_{1,3}
    d21 = d(1, 2, data)[1]    # d_{3-2,3-1} = d_{1,2}

    @inbounds X[1:3,1] .= 0.0
    @inbounds X[1:3,2] .= [-d21; 0.0; 0.0]

    cosθ3 = cosθ(d10, d20, d21)

    @inbounds X[1:3,3] .= [
        d10*cosθ3 - d21;
        d10*sqrt(1.0 - cosθ3^2);
        0.0
    ]

    return X
end

# partial LDE
function compute_partial_lde(l, data::DATA, X, adj)
    partial_lde = 0.0
    @inbounds @views for k in adj[l]
        dist = d(l, data.Dij[k,1], X)
        L,U = data.D[k,1:2]
        if L == U
            partial_lde = max(partial_lde, abs(L - dist)/L)
        else
            partial_lde = max(partial_lde, max( max(L - dist, 0.0)/L, max(dist - U, 0.0)/U ) )
        end
    end
    return partial_lde
end

function sort_torsion_angle(l, data::DATA, sgn)
    # P[l,4] * torsion[l,1] is the center of the interval;
    # torsion[l,2] is the shift from center, so the interval has length 2*torsion[l,2]

    s = sgn * data.torsions[l,1]

    if data.torsions[l,2] == 0.0
        out = s
    else
        # sort an angle in the negative or positive part of the torsion interval
        L = s - data.torsions[l,2]
        U = s + data.torsions[l,2]
        if sgn > 0.0
            L = max(L, 0.0)
        elseif sgn < 0.0
            U = min(U, 0.0)
        end
        out = L + (U - L) * rand()
    end

    return out
end
