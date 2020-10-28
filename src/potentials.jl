import Base: length, +
export Reid


abstract type Potential end
struct Reid <: Potential end

struct Yukawa <: Potential
    V₀::Float64
    η::Float64
end

struct SquareWell <: Potential
    r::Float64
    V0::Float64
end
SquareWell(V0::Float64) = SquareWell(1.0, V0)

length(::Potential) = 0


#=
Reid potential in momentum basis
=#
function (P::Reid)(k, k′)
    exponents = [1 4 7]
    V = [-10.463 -1650.6 6484.3] ./197 # MeV
    V = sum(P.(V, exponents, k, k′))
end

function (P::Reid)(V, η, k, k′)
    μ = 0.7 # fm^-1
    ln = log(((μ*η)^2 + (k + k′)^2)/
             ((μ*η)^2 + (k - k′)^2))
    V/(4μ*k*k′)*ln
end

#=
Reid potential in position basis
=#
function (V::Reid)(r)
    exponents = [1 4 7]
    coeffs = [-10.463 -1650.6 6484.3] ./197 # MeV
    sum(V.(coeffs, exponents, r))
end

function (V::Reid)(coeff, η, r)
    μ = 0.7 # fm^-1
    coeff*exp(-μ*η*r)/(μ*r)
end

#=
Yukawa potential in position basis
=#
function (V::Yukawa)(r)
    V.V₀*exp(-V.η*r)/r
end

#=
Yukawa potential in momentum basis
=#

function (V::Yukawa)(k, k′)
    ln = log(((V.η)^2 + (k + k′)^2)/
             ((V.η)^2 + (k - k′)^2))
    V.V₀/(4k*k′)*ln
end

#=
Square Well potential in position basis
=#
function (P::SquareWell)(r)
    if 0 ≤ r ≤ P.r
        -P.V0
    else
        zero(P.V0)
    end
end

#=
    Analytical phase shifts δ of the square well potential
=#

function analytical(V::SquareWell, mass, E)
    @. atan(√(E/(E+V.V0)) * tan(V.r*√(2mass*(E+V.V0)))) - V.r*√(2mass * E)
end

function analyticalsmooth(V::SquareWell, mass, E; space::Symbol = :E)
    if space == :k
        E = @. E^2/2mass
        atan.(tan.(analytical(V, mass, E)))
    elseif space == :E
        atan.(tan.(analytical(V, mass, E)))
    else
        throw(ArgumentError("Expected `:k` or `:E`"))
    end
end

#= EFT Potentials =#
#= Lowest leading order pionless EFT potential =#
struct LO <: Potential
    C₀::Float64
end

function (V::LO)(k, k′)
    V.C₀
end

#= Next leading order pionless EFT potential =#
struct NLO <: Potential
    C₀::Float64
    C₂::Float64
end

function (V::NLO)(k, k′)
    V.C₀ + V.C₂*(k^2 + k′^2)
end

#= Next next leading order pionless EFT potential =#
struct NNLO <: Potential
    C₀::Float64
    C₂::Float64
    C₄::Float64
end

function (V::NNLO)(k, k′)
    V.C₀ + V.C₂*(k^2 + k′^2) + V.C₄*(k^4 + k′^4) + V.C₄*k^2*k′^2
end

#= Pion interaction =#
struct Pion <: Potential
    mπ::Float64
    Vπ::Float64
end
Pion(Vπ::Real) = Pion(0.7, Vπ)

function (V::Pion)(k, k′)
    mπ = V.mπ
    V.Vπ/(4mπ * k*k′) * log((mπ^2 + (k+k′)^2)/(mπ^2 + (k-k′)^2))
end

struct CompoundPotential <: Potential
    V1::Potential
    V2::Potential
end
function(V::CompoundPotential)(r)
    V.V1(r) + V.V2(r)
end
function(V::CompoundPotential)(k, k′)
    V.V1(k, k′) + V.V2(k, k′)
end

+(V1::Potential, V2::Potential)::Potential = CompoundPotential(V1, V2)

#= UV Regulatization =#
struct UVRegulator <: Potential
    V::Potential
    Λ::Float64  # Cutoff parameter [fm⁻¹]
end

function (V::UVRegulator)(k, k′)
    exp(-k^4/V.Λ^4) * V.V(k, k′) * exp(-k′^4/V.Λ^4)
end

function regularize(V::Potential, Λ; method::Symbol = :UV)::UVRegulator
    UVRegulator(V, Λ)
end
