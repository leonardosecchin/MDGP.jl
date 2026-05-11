# workspace for SPG. V is the point/vector structure
struct SPG_WORKSPACE{V}
    idxD::AbstractVector{Int64}
    w::Vector{Float64}
    dists::Vector{Float64}
    x::V
    g::V
    d::V
    xnew::V
    gnew::V
    s::V
    y::V
    xbest::V
    work::Vector{Float64}
    lastσ::Vector{Float64}
end

function init_spg_workspace(idxD, data::DATA, par::MDGP_PARAMETERS)
    nv = data.nv
    nd = data.nd
    V() = SPGX_VECTOR(Matrix{Float64}(undef,3,nv), Vector{Float64}(undef,nd))
    return SPG_WORKSPACE(
        consec_range(idxD),
        Vector{Float64}(undef,nd),   # w
        Vector{Float64}(undef,nd),   # dists
        V(),   # x
        V(),   # g
        V(),   # d
        V(),   # xnew
        V(),   # gnew
        V(),   # s
        V(),   # y
        V(),   # xbest
        Vector{Float64}(undef,nv),   # work
        fill(-Inf, par.spg_lsm)      # lastσ
    )
end

# Main function
function spg(
    data::DATA, par::MDGP_PARAMETERS, maxit, work::SPG_WORKSPACE, verbose;
    # stress and its gradient
    f::Function = stress,
    g!::Function = grad_stress!,
    # projected direction
    proj_d!::Function = proj_d!,
    # basic operations on vector structure
    cp!::Function = SPGX_VECTOR_cp!,
    xdoty::Function = SPGX_VECTOR_dot,
    xpay!::Function = SPGX_VECTOR_XpaY!,
    supn::Function = SPGX_VECTOR_supn
)
    # Note: in this function, the initial point is considered feasible w.r.t
    # box constraints

    # stress at the initial point
    σ = f(data, work.x, work)

    # stress gradient at the initial point
    g!(data, work.x, work.g, work)

    # stress history
    work.lastσ .= -Inf
    max_σ = σ

    # save initial solution as the best
    σbest = σ
    cp!(work.xbest, work.x, work)

    # initial spectral steplength
    tsmall = max(1e-7 * supn(work.x, work), 1e-10)
    xpay!(work.xnew, work.x, -tsmall, work.g, work)
    g!(data, work.xnew, work.gnew, work)

    xpay!(work.s, work.xnew, -1.0, work.x, work)
    xpay!(work.y, work.gnew, -1.0, work.g, work)

    ss = xdoty(work.s, work.s, work)
    sy = xdoty(work.s, work.y, work)

    lambda = (sy <= 0.0) ? par.spg_lmax : min( par.spg_lmax, max(par.spg_lmin, ss/sy) )

    iter_lack = 0
    iter = 0
    status = 9

    t = Inf
    tmin = 1e-5

    # main loop
    while (true)

        (verbose > 2) && print_info(iter, σ, false, t)

        # test whether a solution was found
        mde, lde = MDE_LDE(data, 1:data.nd, work.dists)
        if (σ <= par.tol_stress) || (mde <= par.tol_mde) || (lde <= par.tol_lde)
            (verbose > 2) && print_info(iter, σ, true, t)

            σbest = σ
            status = 0
            break
        end

        if (iter >= maxit)
            (verbose > 2) && print_info(iter, σ, true, t)

            cp!(work.x, work.xbest, work)
            status = 3
            break
        end

        # ITERATE
        prev_σ = σ

        # projected direction work.d = proj(x - lambda*g) - x
        proj_d!(data, lambda, work)

        # line search
        t = 1.0

        # first trial x + d
        xpay!(work.xnew, work.x, 1.0, work.d, work)
        σ = f(data, work.xnew, work)

        # g'*d
        gtd = xdoty(work.g, work.d, work)

        while (σ > max_σ + t*par.spg_eta*gtd) && (t > tmin)
            if t <= 0.1
                t /= 2.0
            else
                # quadratic interpolation
                tquad = -0.5*( gtd*(t^2) / (σ - prev_σ - t*gtd) )

                if (tquad >= 0.1) && (tquad <= 0.9*t)
                    t = tquad
                else
                    # backtracking
                    t /= 2.0
                end
            end

            # new trial
            xpay!(work.xnew, work.x, t, work.d, work)
            σ = f(data, work.xnew, work)
        end

        # steplength is too small, no progress can be expected
        if t <= tmin
            (verbose > 2) && print_info(iter, σ, true, t)

            cp!(work.x, work.xbest, work)
            status = 4
            break
        else
            # gradient at Znew
            g!(data, work.xnew, work.gnew, work)

            # new spectral steplength
            xpay!(work.s, work.xnew, -1.0, work.x, work)
            xpay!(work.y, work.gnew, -1.0, work.g, work)

            ss = xdoty(work.s, work.s, work)
            sy = xdoty(work.s, work.y, work)

            lambda = (sy <= 0.0) ? par.spg_lmax : min( par.spg_lmax, max(par.spg_lmin, ss/sy) )

            # x = xnew, g = gnew
            cp!(work.x, work.xnew, work)
            cp!(work.g, work.gnew, work)
        end

        # save the new stress value to the history
        @inbounds work.lastσ[mod(iter+1,par.spg_lsm) + 1] = σ

        max_σ = maximum(work.lastσ)

        # best iterate found so far
        if σ < σbest
            σbest = σ
            cp!(work.xbest, work.x, work)
        end

        # lack of progress?
        if par.spg_lacktol > 0.0
            if ((max_σ - σ)/max(1.0, max_σ) > par.spg_lacktol)
                iter_lack = 0
            else
                iter_lack += 1
            end

            if (iter_lack >= max(100, par.spg_lsm))
                (verbose > 2) && print_info(iter, σ, true, t)

                status = 2
                break
            end
        end

        iter += 1
    end

    return σbest, iter, status
end

# print iterate information
function print_info(iter, σ, final, t)
    if (mod(iter,20000) == 0) && !(final)
        println()
        println("  SPG it |     stress | steplength |")
        println(" -----------------------------------")
    end
    if (mod(iter,1000) == 0) || (final)
        @printf(" %7d | %10.3e | %10.3e |\n", iter, σ, t)
    end
end
