###############################
# SPG for conformal coordinates
###############################
include("spg_core.jl")

# point/vector structure
struct SPGX_VECTOR
    X::Matrix{Float64}
    d::Vector{Float64}
end


# Operations between SPGX_VECTORs

# Z = X
function SPGX_VECTOR_cp!(Z::SPGX_VECTOR, X::SPGX_VECTOR, work::SPG_WORKSPACE)
    Z.X .= X.X
    Z.d .= X.d
end

# dot(X,Y)
function SPGX_VECTOR_dot(X::SPGX_VECTOR, Y::SPGX_VECTOR, work::SPG_WORKSPACE)
    r = dot(X.X, Y.X)
    @inbounds for k in work.idxD
        r += X.d[k] * Y.d[k]
    end
    return r
end

# Z = X + a*Y
function SPGX_VECTOR_XpaY!(
    Z::SPGX_VECTOR, X::SPGX_VECTOR, a::Float64, Y::SPGX_VECTOR, work::SPG_WORKSPACE
    )
    @. Z.X = X.X + a * Y.X
    @inbounds for k in work.idxD
        Z.d[k] = X.d[k] + a * Y.d[k]
    end
end

# supnorm(X)
function SPGX_VECTOR_supn(X::SPGX_VECTOR, work::SPG_WORKSPACE)
    r = norm(X.X, Inf)
    @inbounds for k in work.idxD
        r = max(r, abs(X.d[k]))
    end
    return r
end

# projected direction
function proj_d!(data::DATA, lambda::Float64, work::SPG_WORKSPACE)
    @. work.d.X = -lambda * work.g.X
    @inbounds for k in work.idxD
        work.d.d[k] = -lambda * work.g.d[k]
    end

    @inbounds @views for k in work.idxD
        dk = work.d.d[k]
        xk = work.x.d[k]
        aux = xk + dk

        L,U = data.D[k,1:2]
        if aux < L
            work.d.d[k] = L - xk
            elseif aux > U
            work.d.d[k] = U - xk
        end
    end
end


# Stress function
#
# stress(x) = 0.5 * sum_{k-th distance} w[k] * (||x.X_i - x.X_j|| - x.d[k])^2
#
# where i,j are the vertices corresponding to the k-th distance
function stress(
    data::DATA,
    x::SPGX_VECTOR,
    work::SPG_WORKSPACE
    )
    sig = 0.0

    @inbounds @views for k in work.idxD
        i,j = data.Dij[k,1:2]
        work.dists[k] = euclidean(x.X[1:3,i], x.X[1:3,j])
        sig += work.w[k] * (work.dists[k] - x.d[k])^2
        if isnan(sig)
            @show data.Dij[i,j]
            @show work.w[k]
            @show x.d[k]
            @show work.dists[k]
            return
        end
    end

    return sig/2.0
end


# gradient of stress
function grad_stress!(
    data::DATA,
    x::SPGX_VECTOR,
    g::SPGX_VECTOR,
    work::SPG_WORKSPACE
    )
    work.work .= 0.0
    g.X .= 0.0

    @inbounds @views for k in work.idxD
        wk = work.w[k]
        dk = x.d[k]
        distk = work.dists[k]

        if distk > 0.0
            i,j = data.Dij[k,1:2]
            aux = wk * (1.0 - dk/distk)
            work.work[i] += aux
            work.work[j] += aux
            @. g.X[1:3,i] -= aux * x.X[1:3,j]
            @. g.X[1:3,j] -= aux * x.X[1:3,i]
        end

        # gradient w.r.t. d
        g.d[k] = wk * (dk - distk)
    end

    @inbounds @views for i in 1:data.nv
        @. g.X[1:3,i] += work.work[i] * x.X[1:3,i]
    end

    return
end



#TODO: avaliar se definir entrada a entrada é + eficiente
# B
# function B!(B, θ, ω, d)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     @inbounds @views begin
#     B[1:5,1] .= [-cosθ;  sinθ*cosω;  sinθ*sinω; 0.0; d]
#     B[1:5,2] .= [-sinθ; -cosθ*cosω; -cosθ*sinω; 0.0; 0.0]
#     B[1:5,3] .= [0.0; -sinω; cosω; 0.0; 0.0]
#     B[1:5,4] .= [-d*cosθ; d*sinθ*cosω; d*sinθ*sinω; 1.0; d^2/2.0]
#     B[1:5,5] .= [0.0; 0.0; 0.0; 0.0; 1.0]
#     end
#     return
# end
#
# # B^{-1}
# function invB!(B, θ, ω, d)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     @inbounds @views begin
#     B[1:5,1] .= [-cosθ;  -sinθ;  0.0; 0.0; d*cosθ]
#     B[1:5,2] .= [sinθ*cosω; -cosθ*cosω; -sinω; 0.0; -d*sinθ*cosω]
#     B[1:5,3] .= [sinθ*sinω; -cosθ*sinω; cosω; 0.0; -d*sinθ*sinω]
#     B[1:5,4] .= [-d; 0.0; 0.0; 1.0; d^2/2.0]
#     B[1:5,5] .= [0.0; 0.0; 0.0; 0.0; 1.0]
#     end
#     return
# end

# derive B w.r.t. θ
# function dB_θ!(dB, θ, ω, d)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     @inbounds @views begin
#     dB[1:5,1] .= [ sinθ; cosθ*cosω; cosθ*sinω; 0.0; 0.0]
#     dB[1:5,2] .= [-cosθ; sinθ*cosω; sinθ*sinω; 0.0; 0.0]
#     dB[1:5,3] .= 0.0
#     dB[1:5,4] .= [d*sinθ; d*cosθ*cosω; d*cosθ*sinω; 0.0; 0.0]
#     dB[1:5,5] .= 0.0
#     end
#     return
# end

# derive B w.r.t. ω
# function dB_ω!(dB, θ, ω, d)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     @inbounds @views begin
#     dB[1:5,1] .= [0.0; -sinθ*sinω;  sinθ*cosω; 0.0; 0.0]
#     dB[1:5,2] .= [0.0;  cosθ*sinω; -cosθ*cosω; 0.0; 0.0]
#     dB[1:5,3] .= [0.0; -cosω; -sinω; 0.0; 0.0]
#     dB[1:5,4] .= [0.0; -d*sinθ*sinω; d*sinθ*cosω; 0.0; 0.0]
#     dB[1:5,5] .= 0.0
#     end
#     return
# end

# v <- eInf' * B
function eInft_times_B!(v, d)
    v .= [d/2; 0.0; 0.0; 0.5 - d^2/4; 0.5]
end

# v <- B * e0
# function B_times_e0!(v, θ, ω, d)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     v .= [-d*cosθ; d*sinθ*cosω; d*sinθ*sinω; 1.0; 1.0 + d^2/2.0]
# end

# v <- B * v
# function B_times_v!(v, θ, ω, d, work)
#     sinθ, cosθ = sincos(θ)
#     sinω, cosω = sincos(ω)
#     @inbounds begin
#     work[1] = -v[1]*cosθ - v[2]*sinθ - v[4]*cosθ
#     work[2] = v[1]*sinθ*cosω - v[2]*cosθ*cosω - v[3]*sinω + v[4]*d*sinθ*cosω
#     work[3] = v[1]*sinθ*sinω - v[2]*cosθ*sinω + v[3]*cosω + v[4]*d*sinθ*sinω
#     work[4] = v[4]
#     work[5] = v[1]*d + v[4]*d^2/2
#     end
#     v .= work
# end

# v <- v' * B
function vt_times_B!(v, θ, ω, d, work)
    sinθ, cosθ = sincos(θ)
    sinω, cosω = sincos(ω)
    @inbounds begin
    work[1] = -v[1]*cosθ + v[2]*sinθ*cosω + v[3]*sinθ*sinω + v[5]*d
    work[2] = -v[1]*sinθ - v[2]*cosθ*cosω - v[3]*cosθ*sinω
    work[3] = -v[2]*sinω + v[3]*cosω
    work[4] = -v[1]*d*cosθ + v[2]*d*sinθ*cosω + v[3]*d*sinθ*sinω + v[4] + v[5]*d^2/2
    work[5] = v[5]
    end
    v .= work
end

# v <- v' * ∂B/∂θ
function vt_times_∂B∂θ!(v, θ, ω, d, work)
    sinθ, cosθ = sincos(θ)
    sinω, cosω = sincos(ω)
    @inbounds begin
    work[1] = v[1]*sinθ + v[2]*cosθ*cosω + v[3]*cosθ*sinω
    work[2] = -v[1]*cosθ + v[2]*sinθ*cosω + v[3]*sinθ*sinω
    work[3] = 0.0
    work[4] = v[1]*d*sinθ + v[2]*d*cosθ*cosω + v[3]*d*cosθ*sinω
    work[5] = 0.0
    end
    v .= work
end

# v <- v' * ∂B/∂ω
function vt_times_∂B∂ω!(v, θ, ω, d, work)
    sinθ, cosθ = sincos(θ)
    sinω, cosω = sincos(ω)
    @inbounds begin
    work[1] = -v[2]*sinθ*sinω + v[3]*sinθ*cosω
    work[2] = -v[2]*cosθ*sinω - v[3]*cosθ*cosω
    work[3] = -v[2]*cosω - v[3]*sinω
    work[4] = v[2]*d*sinθ*sinω + v[3]*d*sinθ*cosω
    work[5] = 0.0
    end
    v .= work
end

# Stress function
# ds: known exact distances from x_{i-1} to x_i (fixed, copied from D)
# θs, ωs: planar and torsion angles (variables)
# Zd: distances in the optimization model (variables)
function stress(
    data::DATA,
    θs, ωs, Zd,
    work::WORKSPACE
)
    sig = 0.0
    u = Vector{Float64}(undef, 5)
    aux = Vector{Float64}(undef, 5)

    @inbounds for v in 1:data.nd
        i,j = data.Dij[v,1:2]
        #TODO: accumulate and reuse products eInf' * Bi *...* Bk
        eInft_times_B!(u, data.preddists[i+1])
        for k in (i+2):j
            vt_times_B!(u, θs[k], ωs[k], data.preddists[k], aux)
        end
        # add w * (|xi - xj|^2 - d^2) ^ 2
        # here, |xi - xj|^2 = 2 * (u' * e0)
        sig += work.w[v] * ( 2 * (u[4] + u[5]) - Zd[v]^2 )^2
    end
    if isnan(sig)
        error("stress returned NaN")
    end

    return sig/2.0
end


# gradient of stress
function grad_stress(
    data::DATA,
    θs, ωs, Zd,
    work::WORKSPACE
    Gθ::Vector{Float64},
    Gω::Vector{Float64}
)
    work.work .= 0.0
    Gθ .= 0.0
    Gω .= 0.0

    u = Vector{Float64}(undef, 5)
    aux = Vector{Float64}(undef, 5)

    @inbounds for v in 1:data.nd
        i,j = data.Dij[v,1:2]
        eInft_times_B!(u, data.preddists[i+1])
        for k in (i+2):j
            @views work.Gθwork[1:5,k] .= u
            @views work.Gωwork[1:5,k] .= u
            vt_times_∂B∂θ(
                @views work.Gθwork[1:5,k], θs[k], ωs[k], data.preddists[k], aux
            )
            vt_times_∂B∂ω(
                @views work.Gωwork[1:5,k], θs[k], ωs[k], data.preddists[k], aux
            )
            vt_times_B!(u, θs[k], ωs[k], data.preddists[k], aux)
        end
        # w * (|xi - xj|^2 - Zd^2) = w * (2 * (u' * e0) - Zd^2)
        wZd = work.w[v] * (2 * (u[4] + u[5]) - Zd[v]^2)

        # gradient w.r.t. angles (already multiplied by 0.5)
        for k in (i+2):j
            Gθ[k] = wZd * (work.Gθwork[4,k] + work.Gθwork[5,k])
            Gω[k] = wZd * (work.Gωwork[4,k] + work.Gωwork[5,k])
        end

        # gradient w.r.t. d (already multiplied by 0.5)
        Gd[k] = -wZd * Zd[v]
    end

    return
end
