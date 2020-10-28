using ArgCheck
using UnitfulIntegration
using LinearAlgebra
import Unitful: fm
import QuadGK
using .Scattering

struct KMatrix <: Method
    N::Int  # Number of iteration steps
end

function (method::KMatrix)(k₀, m, P::Potential)
    A, V = createA(k₀, m, method.N, P)
    # Create the reactance matrix K using the matrix equation AK = V
    K = inv(A)*V
    @assert size(A) == size(V) == size(K) == (method.N+1, method.N+1)

    # Diagonal elements of K are related to the phase shifts δₗ
    # as K(k₀, k₀) = -tan(δₗ)/mk₀
    Kk0 = K[end, end]
    δᵢ = atan(-Kk0*m*k₀) / pi * 180
    #energy = k₀^2 / m * 197
    δᵢ
end

function createA(k₀, m, N::Int, P::Potential)
    k, ω = QuadGK.gauss(N)
    transform!(k, ω)
    @assert k₀ ∉ k "k₀ can not be in the mesh points k"
    push!(k, k₀)
    V = [P(ki, kj) for ki in k, kj in k]
    A = createA(V, k, ω, m)
    return A, V
end

function createA(V, k, ω, m)
    @argcheck length(k) - 1 == length(ω)
    @argcheck size(V, 1) == size(V, 2) == length(k)
    @argcheck m > 0

    N = length(k)-1
    k₀ = k[end]
    A = diagm(0 => repeat([1.0,], N+1))  # δᵢⱼ
    uⱼ = 0.0
    @inbounds for j in 1:N+1
        uⱼ = 0.0
        if j ≠ N+1
            uⱼ = 2/π * ω[j]*k[j]^2/((k₀^2 - k[j]^2)/m)
        else
            for n in 1:N
                uⱼ += ω[n]*k₀^2/((k₀^2 - k[n]^2)/m)
            end
            uⱼ *= -2/π
        end

        for i in 1:N+1
            A[i, j] -= V[i, j] * uⱼ
        end
    end
    return A
end

function transform!(xs, weights)
    @. weights = π/4 * weights / cos(π/4*(1.0 + xs))^2
    @. xs      = tan(π/4*(1.0 + xs))
end

