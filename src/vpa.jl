#using Libdl
#using Unitful
using DifferentialEquations
using .Scattering: Method, Potential

struct VPA <: Method
    ρ::Float64  # Radius for where to truncate the potential
    rspan::Tuple{Float64, Float64}  # Range of the interaction
end
VPA() = VPA(1.0, (1e-4, 15.0))
VPA(ρ::Real) = VPA(ρ, (1e-4, 15.0))
VPA(rspan::Tuple{Float64, Float64}) = VPA(1.0, rspan)


function (vpa::VPA)(k, m, V::Potential;kw...)
    #Vρ(r) = if r ≤ vpa.ρ V(r) else 0 end
    #m = m/2

    function dδdr(δ, p, r)
        -1.0/k * 2m * V(r) * sin(k*r + δ)^2
    end

    δ0 = 0.0

    prob = ODEProblem(dδdr, δ0, vpa.rspan)
    sol = solve(prob, reltol=1e-8; kw...)

    δ = sol.u[end]
    #kcot = k/tan(δ)
    #δ/ pi * 180
    δ
end
