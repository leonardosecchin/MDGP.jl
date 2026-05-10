mutable struct MDGP_PARAMETERS
    N_sols::Int64       # number of solutions required
    N_trial::Int64      # max number of initial trials
    N_conf::Int64       # number of initial conformations
    N_impr::Int64       # number of improvement trials
    N_tors::Int64       # max number of torsion angles trials
    N_similar::Int64    # max number of consecutive similar init conf
    # tolerances
    tol_lde::Real       # tolerance for optimality (LDE)
    tol_mde::Real       # tolerance for optimality (MDE)
    tol_stress::Real    # tolerance for optimality (SPG, stress)
    tol_exact::Real     # maximum interval length to consider a distance exact
    tol_similar::Real   # tolerance to consider two conformations similar
    # SPG options
    spg_maxit::Int64    # maximum number of SPG iterations
    spg_lacktol::Real   # tolerance to declare lack of progress in SPG
    spg_eta::Real       # Armijo's parameter
    spg_lsm::Int64      # length of the history for non-monotone line search
    spg_lmin::Real      # minimum value for spectral steplength
    spg_lmax::Real      # maximum value for spectral steplength
    # other
    tight_bounds::Bool  # try tightening the distance bounds
    max_time::Real      # max time in seconds
end

"""
    par = mdgp_default_parameters()

Returns a structure MDGP_PARAMETERS with default values. See `mdgp_multistart`
help for more details.
"""
function mdgp_default_parameters()
    return MDGP_PARAMETERS(
        1,        # number of solutions required
        500,      # max number of initial trials
        50,       # number of initial conformations
        3,        # number of improvement trials
        20,       # max number of torsion angles trials
        50,       # max number of consecutive similar init conf
        # tolerances
        1e-2,     # tolerance for optimality (LDE)
        1e-3,     # tolerance for optimality (MDE)
        1e-7,     # tolerance for optimality (SPG, stress)
        1e-12,    # maximum interval length to consider a distance exact
        5.0,      # tolerance to consider two conformations similar
        # SPG options
        30000,    # maximum number of SPG iterations
        1e-8,     # tolerance to declare lack of progress in SPG
        1e-4,     # Armijo's parameter
        10,       # length of the history for non-monotone line search
        1e-20,    # minimum value for spectral steplength
        1e+20,    # maximum value for spectral steplength
        # other
        true,     # try tightening the distance bounds
        7200      # max time in seconds
    )
end
