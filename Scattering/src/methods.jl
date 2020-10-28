using .Scattering
import Base: length

abstract type Method end
phaseshift(k₀::Real, m, P::Potential, method::Method; kwargs...) = method(k₀, m, P; kwargs...)
phaseshift(k₀::AbstractArray, m, P::Potential, method::Method; kwargs...) = [method(k, m, P; kwargs...) for k in k₀]
length(::Method) = 1
crossection(δₗ::Real, k::Real) = 4π*sin(δₗ)^2/k^2
crossection(δₗ::Vector{<:Real}, k::Vector{<:Real}) = @. 4π*sin(δₗ)^2/k^2
