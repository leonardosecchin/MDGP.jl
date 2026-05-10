"""
Multistart strategy for Molecular Distance Geometry Problem
"""

module MDGP

using LinearAlgebra
using SparseArrays
using Printf
using Distances
using Random
using Distributions
using DelimitedFiles

export MDGP_PARAMETERS, mdgp_default_parameters
export mdgp_multistart, mdgp_read

include("structs.jl")
include("check.jl")
include("vectors.jl")
include("basic.jl")
include("preprocess.jl")
include("spg.jl")
include("conformation.jl")
include("read.jl")

########################
# Main function
########################
"""
    sols, types, ldes, mdes, time_init, time_total = mdgp_multistart(Dij, D, P, atoms, torsions; par, [OPTIONS])

The multistart strategy for MDGP described in

`Secchin, da Rocha, da Rosa, Liberti, Lavor. A fast heuristic for the molecular distance geometry problem. 2025`

The MDGP instance data `Dij`, `D`, `P`, `atoms` and `torsions` must be provided.
See `mdgp_read` help.

## Output
- `sols`: list of conformations computed
- `types`: vector of status of each conformation in `sols`
- `-1`: infeasible
- `1`: feasible conformation computed before applying SPG
- `2`: feasible conformation computed after applying SPG
- `ldes`: LDE of each conformation in `sols`
- `mdes`: MDE of each conformation in `sols`
- `time_init`: preprocessing time
- `time_total`: total time

## Options

- `seed`: random seed (`<0` for any) (`-1`)
- `verbose`: output level (`0` none, `1` normal, `2,3` detailed) (`1`)

## Parameters

Parameters are passed through input `par`, which is a `MDGP_PARAMETERS` structure.
If not provided, it is initialized by the function `mdgp_default_parameters()`,
which sets the default values as listed below. To set different values, first
initialize the `MDGP_PARAMETERS` structure:

`par = mdgp_default_parameters()`

Then, you can modify it. For example, to set the number of initial trials
to 1000, run `par.N_trial = 1000`. The complete list of parameters and their
default values is given below.

### Generation of conformations

- `N_sols`: number of solutions required (`1`)
- `N_trial`: max number of initial trials (`500`)
- `N_conf`: number of initial conformations (`50`)
- `N_impr`: number of improvement trials (`3`)
- `N_tors`: max number of torsion angles trials (`20`)
- `N_similar`: max number of consecutive similar initial conformations (`50`)

### Tolerances

- `tol_lde`: tolerance for optimality (LDE) (`1e-2`)
- `tol_mde`: tolerance for optimality (MDE) (`1e-3`)
- `tol_stress`: tolerance for optimality (SPG, stress) (`1e-7`)
- `tol_exact`: maximum interval length to consider a distance exact (`1e-12`)
- `tol_similar`: tolerance to consider two conformations similar (`5.0`)

### SPG options

- `spg_maxit`: maximum number of SPG iterations equals to max(spg_maxit, 20*#atoms) (`30000`)
- `spg_lacktol`: tolerance to declare lack of progress (`1e-8`)
- `spg_eta`: Armijo's parameter (`1e-4`)
- `spg_lsm`: length of the history for non-monotone line search (`10`)
- `spg_lmin`: minimum value for spectral steplength (`1e-20`)
- `spg_lmax`: maximum value for spectral steplength (`1e+20`)

### Other

- `tight_bounds`: try tightening the distance bounds (`true`)
- `max_time`: maximum time in seconds (`7200`)
"""
function mdgp_multistart(
    Dij_orig::Matrix{Int64},         # nd x 2 matrix of indices of distances
    D_orig::Matrix{Float64},         # nd x 2 matrix of distances
    P_orig::Matrix{Int64},           # predecessors and branching signs
    atoms::Vector{String},           # atoms names
    tor_orig::Matrix{Float64};       # torsion angles and left/right displacements (in degrees)
    par::MDGP_PARAMETERS = mdgp_default_parameters(),
    seed::Int64          = -1,       # random seed (<0 for any)
    verbose::Int64       = 1         # output level
)

    # check parameters
    check_param(par.N_conf > 0, "N_conf must be positive")
    check_param(par.N_trial > 0, "N_trial must be positive")
    check_param(par.N_tors >= 0, "N_tors must be non negative")
    check_param(par.N_sols > 0, "N_sols must be positive")
    check_param(par.max_time > 0, "max_time must be positive")

    check_param(par.spg_lsm > 0, "spg_lsm must be positive")
    check_param((par.spg_eta > 0) && (par.spg_eta < 1.0), "spg_eta must be in (0,1)")
    check_param(par.spg_lmin > 0, "spg_lmin must be positive")
    check_param(par.spg_lmax > 0, "spg_lmax must be positive")

    check_param((par.tol_mde > 0) || (par.tol_lde > 0), "tol_mde or tol_lde must be positive")
    check_param(par.tol_exact >= 0, "tol_exact must be positive")
    check_param(par.tol_similar >= 0, "tol_similar must be non negative")

    if seed >= 0
        Random.seed!(seed)
    end

    time_pre = @elapsed begin

    # problem data and preprocess
    data = init_problem_data(Dij_orig, D_orig, P_orig, tor_orig)

    # check the existence of necessary distances
    check_necessarydistances(data, par)

    # try to improve inexact distances
    if par.tight_bounds
        infeas = tight_bounds!(data, par, verbose)
        check(!infeas, "After tightening the bounds, an infeasibility was found. Run with 'tight_bounds=false' to ignore it.")
    end
    check(any(data.D[:,2] .> data.D[:,1]), "Instance is exact, enumerative strategies are recommended")

    # groups of distances
    idxDpred = Int64[]      # between predecessors
    idxDnonpred = Int64[]   # extra
    idxDvdw = Int64[]       # Van der Walls
    @inbounds for k in 1:size(data.D,1)
        i,j = data.Dij[k,1:2]
        if (j in data.P[i,1:3]) || (i in data.P[j,1:3])
            push!(idxDpred, k)
        elseif data.D[k,2] < 900.0
            push!(idxDnonpred, k)
        else
            push!(idxDvdw, k)
        end
    end

    idxDpred = consec_range(idxDpred)
    idxDnonpred = consec_range(idxDnonpred)
    idxDvdw = consec_range(idxDvdw)

    idxDprednonpred = consec_range(union(idxDpred,idxDnonpred))

    # adjacent distances to each atom
    adj = Vector{Vector{Int64}}(undef, data.nv)
    for v in 1:data.nv
        adj[v] = Int64[]
        for k in 1:data.nd
            if data.Dij[k,2] == v
                push!(adj[v], k)
            end
        end
    end

    # ======================
    # First definitions and allocation
    # ======================

    X = start_conformation(data)

    # initialize workspace
    spg_work = init_spg_workspace(data, par)

    # stress weights
    spg_work.w .= 1.0
    @views spg_work.w[idxDpred] .= 2.0      # discretization distances
    spg_work.w ./= norm(spg_work.w)

    stop = false

    total_count = 0

    fixed_torsions = Vector{Float64}(undef, data.nv)
    fixed_torsions[1:3] .= 0.0

    # vector of solutions
    sols = Vector{Matrix{Float64}}(undef, 0)
    ldes = Float64[]
    mdes = Float64[]
    soltypes = Int64[]
    stress = Float64[]

    strs = 0.0

    # similarity between generated conformations
    consec_similar = 0
    if data.nv <= 500
        rmsd_idx = 1:data.nv
    else
        rmsd_idx = (1:data.nv)[atoms .== "CA"]
    end
    Xtmp1 = similar(X)
    Xtmp2 = similar(X)

    end    # end of @elapsed begin

    if verbose > 0
        ndexact = count(data.D[:,2] .- data.D[:,1] .<= par.tol_exact)
        @printf("Time preprocess phase and allocation: %.6lf s\n", time_pre)
        println("Preprocessed data has $(data.nv) atoms, $(ndexact) exact distances and $(data.nd - ndexact) interval distances.")
    end

    # ======================
    # Starting conformations
    # ======================

    if verbose > 0
        println("\nComputing initial conformations...")
    end

    time_total = 0.0
    time_init = 0.0

    while (length(sols) < par.N_conf) && (total_count < par.N_trial)

        time_initk = @elapsed begin

        if verbose > 0
            print("\rAttempt $(total_count), $(length(sols)) conformations found")
        end

        if count(soltypes .>= 0) >= par.N_sols
            stop = true
            break
        end

        # sort torsion angles
        @inbounds for v in 4:data.nv
            sgn = (data.P[v,4] == 0) ? rand([-1,1]) : data.P[v,4]
            fixed_torsions[v] = sort_torsion_angle(v, data, sgn)
        end

        # new conformation
        @inbounds if construct_conformation!(
            4, data, X, fixed_torsions, adj, par.N_tors, par, false
        )

            total_count += 1
            # Try to improve the new conformation by minimizing LDE
            # Note that we can neglect discretization distances as
            # they will still be satisfied.
            if par.N_impr > 0
                improve_conformation!(
                    idxDprednonpred, data, X, fixed_torsions, adj, par
                )
            end

            mde, lde = MDE_LDE(data, X)

            if (lde <= par.tol_lde) || (mde <= par.tol_mde)
                stop = true
            end

            rmsdpass = true
            if !stop
                # different conformation?
                centralize!(X, Xtmp1, rmsd_idx)
                for c in 1:length(sols)
                    centralize!(sols[c], Xtmp2, rmsd_idx)
                    rmsd = rmsd_protein(Xtmp1, Xtmp2, rmsd_idx)
                    if rmsd <= par.tol_similar
                        rmsdpass = false
                        break
                    end
                end
                if rmsdpass
                    consec_similar = 0
                else
                    consec_similar += 1
                    if consec_similar >= par.N_similar
                        # maximum number of consecutive trials with similar conformations
                        stop = true
                    end
                end
            end

            spg_applied = false
            if !stop && rmsdpass
                # different infeasible conformation, try to improve it through SPG
                if verbose > 2
                    println("\n\n>>> Trying to improve conformation through SPG")
                end
                strs, spg_it, st = spg(
                    idxDprednonpred, X, data, par, spg_work, verbose
                )

                mde, lde = MDE_LDE(data, X)

                if (lde <= par.tol_lde) || (mde <= par.tol_mde)
                    stop = true
                end

                spg_applied = true
            end

            if rmsdpass
                push!(sols, deepcopy(X))
                push!(soltypes, -1)
                push!(ldes, lde)
                push!(mdes, mde)
                push!(stress, spg_applied ? strs : Inf)
            end

            if (lde <= par.tol_lde) || (mde <= par.tol_mde)
                # X is feasible!
                soltypes[end] = spg_applied ? 2 : 1

                if verbose > 0
                    if spg_applied
                        println("\rA solution was found after applying SPG")
                    else
                        println("\rA solution was found in the initialization phase")
                    end
                end
            end
        end
        end   # end of @elapsed begin

        time_init += time_initk
        time_total = time_pre + time_init

        if stop || (time_total >= par.max_time)
            break
        end
    end

    if verbose > 0
        println("\r$(length(sols)) initial conformations computed in $(total_count) trials.")

        println("\nSUMMARY\n$(repeat('-',48))")
        @printf("Total time: %.6lf s\n", time_total)
        println("Number of solutions found: $(count(soltypes .>= 0))")
        b = argmin(mdes)
        @printf("Minimum MDE among all conformations: %9.3e\n", mdes[b])
        @printf("LDE of the corresponding conformation: %9.3e\n", ldes[b])
    end

    return sols, soltypes, ldes, mdes, time_init, time_total
end

end
