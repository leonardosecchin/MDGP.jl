function spg(
    idxX, idxD, ZX, data::DATA, par::MDGP_PARAMETERS,
    work::WORKSPACE,
    verbose
)
    # Zd_ij = projection of |Xi - Xj| onto the interval distance
    @inbounds @views for k in idxD
        work.Zd[k] = clamp(work.dists[k], data.D[k,1], data.D[k,2])
    end

    # stress at the initial point
    sig = stress(idxD, data.Dij, work.Zd, work)

    # stress gradient at the initial point
    grad_stress(idxX, idxD, data.Dij, ZX, work.Zd, work.GX, work.Gd, work)

    # stress history
    work.lastf .= -Inf
    max_sig = sig

    # save initial solution as the best
    sigbest = sig
    cp!(idxX, work.ZXbest, ZX)
    cp!(idxD, work.Zdbest, work.Zd)

    # initial spectral steplength
    nXD = max(norm(ZX[idxX], Inf), norm(work.Zd[idxD], Inf))
    tsmall = max(1e-7 * nXD, 1e-10)
    XpaY!(idxX, work.ZXnew, ZX, -tsmall, work.GX)
    XpaY!(idxD, work.Zdnew, work.Zd, -tsmall, work.Gd)
    grad_stress(idxX, idxD, data.Dij, work.ZXnew, work.Zdnew, work.GXnew, work.Gdnew, work)

    XpaY!(idxX, work.SX, work.ZXnew, -1.0, ZX)
    XpaY!(idxX, work.YX, work.GXnew, -1.0, work.GX)

    XpaY!(idxD, work.Sd, work.Zdnew, -1.0, work.Zd)
    XpaY!(idxD, work.Yd, work.Gdnew, -1.0, work.Gd)

    SS = XdotY(idxX, work.SX, work.SX) + XdotY(idxD, work.Sd, work.Sd)
    SY = XdotY(idxX, work.SX, work.YX) + XdotY(idxD, work.Sd, work.Yd)

    lambda = (SY <= 0.0) ? par.spg_lmax : clamp(SS/SY, par.spg_lmin, par.spg_lmax)

    iter_lack = 0
    iter = 0
    status = 9

    t = Inf
    tmin = 1e-5

    # main loop
    while (true)

        if verbose > 2
            print_info(iter, sig, false, t)
        end

        # test whether a solution was found
        mde, lde = MDE_LDE(data, 1:size(data.Dij,1), work.dists)
        if (sig <= par.tol_stress) || (mde <= par.tol_mde) || (lde <= par.tol_lde)
            if verbose > 2
                print_info(iter, sig, true, t)
            end

            sigbest = sig
            status = 0
            break
        end

        if (iter >= max(par.spg_maxit, 20*data.nv))
            if verbose > 2
                print_info(iter, sig, true, t)
            end

            cp!(idxX, ZX, work.ZXbest)
            status = 3
            break
        end

        # ITERATE
        prev_sig = sig

        # projected direction P = proj(Z - lambda*G) - Z
        aX!(idxD, work.Pd, -lambda, work.Gd)

        # project Pd = projection of Zd + Pd onto [dL,dU]
        @inbounds @views for k in idxD
            aux = work.Zd[k] + work.Pd[k]
            L,U = data.D[k,1:2]
            if aux < L
                work.Pd[k] = L - work.Zd[k]
            elseif aux > U
                work.Pd[k] = U - work.Zd[k]
            end
        end

        # line search
        t = 1.0

        XpaY!(idxX, work.ZXnew, ZX, -lambda, work.GX)
        XpY!(idxD, work.Zdnew, work.Zd, work.Pd)
        @views for k in idxD
            i,j = data.Dij[k,1:2]
            work.dists[k] = euclidean(work.ZXnew[1:3,i], work.ZXnew[1:3,j])
        end
        sig = stress(idxD, data.Dij, work.Zdnew, work)

        # inner product <G,P>
        GP = lambda*XdotY(idxX, work.GX, work.GX) + XdotY(idxD, work.Gd, work.Pd)

        while (sig > max_sig + t*par.spg_eta*GP) && (t > tmin)
            if t <= 0.1
                t /= 2.0
            else
                # quadratic interpolation
                tquad = -0.5*( GP*(t^2) / (sig - prev_sig - t*GP) )

                if (tquad >= 0.1) && (tquad <= 0.9*t)
                    t = tquad
                else
                    # backtracking
                    t /= 2.0
                end
            end

            # new trial
            XpaY!(idxX, work.ZXnew, ZX, -t*lambda, work.GX)
            XpaY!(idxD, work.Zdnew, work.Zd,  t, work.Pd)
            @views for k in idxD
                i,j = data.Dij[k,1:2]
                work.dists[k] = euclidean(work.ZXnew[1:3,i], work.ZXnew[1:3,j])
            end
            sig = stress(idxD, data.Dij, work.Zdnew, work)
        end

        # steplength is too small, no progress can be expected
        if t <= tmin
            if verbose > 2
                print_info(iter, sig, true, t)
            end

            cp!(idxX, ZX, work.ZXbest)
            status = 4
            break
        else
            # gradient at Znew
            grad_stress(idxX, idxD, data.Dij, work.ZXnew, work.Zdnew, work.GXnew, work.Gdnew, work)

            # new spectral steplength
            XpaY!(idxX, work.SX, work.ZXnew, -1.0, ZX)
            XpaY!(idxX, work.YX, work.GXnew, -1.0, work.GX)

            XpaY!(idxD, work.Sd, work.Zdnew, -1.0, work.Zd)
            XpaY!(idxD, work.Yd, work.Gdnew, -1.0, work.Gd)

            SS = XdotY(idxX, work.SX, work.SX) + XdotY(idxD, work.Sd, work.Sd)
            SY = XdotY(idxX, work.SX, work.YX) + XdotY(idxD, work.Sd, work.Yd)

            lambda = (SY <= 0.0) ? par.spg_lmax : min( par.spg_lmax, max(par.spg_lmin, SS/SY) )

            # Z = Znew
            cp!(idxX, ZX, work.ZXnew)
            cp!(idxD, work.Zd, work.Zdnew)

            # G = Gnew
            cp!(idxX, work.GX, work.GXnew)
            cp!(idxD, work.Gd, work.Gdnew)
        end

        # save the new stress value to the history
        @inbounds work.lastf[mod(iter+1,par.spg_lsm) + 1] = sig

        max_sig = maximum(work.lastf)

        # best iterate found so far
        if sig < sigbest
            sigbest = sig
            cp!(idxX, work.ZXbest, ZX)
            cp!(idxD, work.Zdbest, work.Zd)
        end

        # lack of progress?
        if par.spg_lacktol > 0.0
            if ((max_sig - sig)/max(1.0, max_sig) > par.spg_lacktol)
                iter_lack = 0
            else
                iter_lack += 1
            end

            if (iter_lack >= max(100, par.spg_lsm))
                if verbose > 2
                    print_info(iter, sig, true, t)
                end

                status = 2
                break
            end
        end

        iter += 1
    end

    return sigbest, iter, status
end


# print iterate information
function print_info(iter, sig, final, t)
    if (mod(iter,20000) == 0) && !(final)
        println()
        println("  SPG it |     stress | steplength |")
        println(" -----------------------------------")
    end
    if (mod(iter,1000) == 0) || (final)
        @printf(" %7d | %10.3e | %10.3e |\n", iter, sig, t)
    end
end


# Stress function
#
# stress(Z) = 0.5 * sum_{k-th distance} w[k] * (||ZX_i - ZX_j|| - Zd[k])^2
#
# where i,j are the vertices corresponding to the k-th distance
function stress(
    idxD::AbstractVector{Int64},
    Dij::Matrix{Int64},
    Zd::Vector{Float64},
    work::WORKSPACE
)

    sig = 0.0

    @inbounds @views for k in idxD
        sig += work.w[k] * (work.dists[k] - Zd[k])^2
        if isnan(sig)
            @show Dij[k,:]
            @show work.w[k]
            @show Zd[k]
            @show work.dists[k]
            return
        end
    end

    return sig/2.0
end


# gradient of stress
function grad_stress(
    idxX::AbstractVector{Int64},
    idxD::AbstractVector{Int64},
    Dij::Matrix{Int64},
    ZX::Matrix{Float64},
    Zd::Vector{Float64},
    GX::Matrix{Float64},
    Gd::Vector{Float64},
    work::WORKSPACE
)

    work.work .= 0.0
    GX .= 0.0

    @inbounds @views for k in idxD
        wk = work.w[k]

        if work.dists[k] > 0.0
            i,j = Dij[k,1:2]
            aux = wk * (1.0 - Zd[k]/work.dists[k])
            work.work[i] += aux
            work.work[j] += aux
            @. GX[1:3,i] -= aux * ZX[1:3,j]
            @. GX[1:3,j] -= aux * ZX[1:3,i]
        end

        # gradient w.r.t. d
        Gd[k] = wk * (Zd[k] - work.dists[k])
    end

    @inbounds @views for i in 1:size(ZX,2)
        @. GX[1:3,i] += work.work[i] * ZX[1:3,i]
    end

    return
end
