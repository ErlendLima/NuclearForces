using Scattering
using Statistics
using Optim
using LineSearches
using BlackBoxOptim
using LsqFit
using ProgressBars

# Reduced mass in lab frame. Default units [fm^-1]
const mass = m_np

# raw = readdlm("../../project1/data/data.csv")
# head, body = Symbol.(raw[1, :]), Float64.(raw[2:end, :])
# data = DataFrame(body, head);
# @show head;

function findjump(arr::AbstractArray)::Bool
    if length(arr) < 2
        return false
    end
    for i in 2:length(arr)
        if arr[i]*arr[i-1] < 0
            return true
        end
    end
    false
end

energytomomentum(e) = √(mass/(2*197)*e)

# Create a function to do the book-keeping for RMatrix calculation and VPA calculation
function computerange(start, stop, length)::Vector{Float64}
    if stop < start
        start = 0.1stop
    end
    #stop /= 2
    #start /= 2
    10 .^(range(log10(start), log10(stop), length=length))
end

function computephase(V::Potential; method::Scattering.Method=KMatrix(30),
                      startpoint=1e-3, endpoint=3, length=50)
    # Logspaced points to introduce a bias in the error function
    # towards the low energy region
    E = computerange(startpoint, endpoint, length)
    k = energytomomentum.(E)
    δ = phaseshift(k, mass, V, method)
    E, δ
    # 2E if error
end
computephase(V::Potential, m::Scattering.Method; kwargs...) = computephase(V; method=m, kwargs...)

# Compute an "exact" phase shift for comparison. This is used to construct a factory function which creates the error function and handles all of the book keeping of creating a potential, inserting the coefficients and computing the mean square error.
MSE(x, y) = mean((x.-y).^2)
# Using MSE doesn't make sense. Use χ²
# maybe relative error? Yes, relative error
χ²(fact, test) = @. (test - fact)^2/fact
Σχ²(fact, test) = sum(χ²(fact, test))
relerr(fact, test) = @. abs(test-fact)/fact
Σrelerr(fact, test) = sum(relerr(fact, test))

function makeerrorfunction(V::Type{<:Potential}; method::Scattering.Method=KMatrix(30), 
                           Λ=0.7, fact=nothing, kwargs...)
    E_fit, δReid_fit = computephase(Reid(); kwargs...);
    k = energytomomentum.(E_fit)
    if isnothing(fact)
        fact = δReid_fit
        factcot = @. k*cot(fact |> deg2rad)
    end
    function error(coeffs)::Float64
        W = regularize(V(coeffs...), Λ)
        _, δ = computephase(W, method; kwargs...)
        if findjump(δ)
            # Compute error with kcot
            d = @. k*cot(δ |> deg2rad)
            return Σrelerr(factcot, d)
        end
        #Σχ²(fact, δ)
        Σrelerr(fact, δ)
    end
    return error
end
makeerrorfunction(V::Type{<:Potential}, method::Scattering.Method; kwargs...) = makeerrorfunction(V; method=method, kwargs...)

struct Params
    start::Float64
    stop::Float64
    length::Int64
    Λ::Vector{Float64}
end
Params(start, stop, length) = Params(start, stop, length, [0.7])
Params(start, stop) = Params(start, stop, 50)
Params(stop) = Params(1e-3, stop)

abstract type AbstractFitParams end

struct FitParams <: AbstractFitParams
    params::Params
    V::Type{<:Potential}
    minlim::Vector{Float64}
    maxlim::Vector{Float64}
    C₀::Vector{Float64}
    method::Symbol
    methodkws::Any
    kwargs::Any
end
function FitParams(params, V::Type{<:Potential}, minlim::Vector{<:Real}, maxlim::Vector{<:Real}, C₀::Vector{<:Real};
        method=:LBFGS, methodkws=Dict(), kwargs...)
   FitParams(params, V, minlim, maxlim, C₀, method, methodkws, kwargs) 
end
function FitParams(params, V::Type{<:Potential}, minlim::Real, maxlim::Real; kwargs...)
    FitParams(params, V, [minlim], [maxlim], Float64[]; kwargs...)
end
function FitParams(params, V::Type{<:Potential}; C₀, kwargs...) 
    minlim = fill(-Inf, length(C₀))
    maxlim = fill(Inf, length(C₀))
    FitParams(params, V, minlim, maxlim, C₀; kwargs...)
end
FitParams(params, V::Type{<:Potential}, minlim::Nothing, maxlim::Nothing; kwargs...) = FitParams(params, V; kwargs...)

struct FitResult
    params::AbstractFitParams
    res::Any
    C::Vector{Float64}
end
FitResult(p::AbstractFitParams, res, C::Float64) = FitResult(p, res, [C])

struct FitParamsPion <: AbstractFitParams
    params::Params
    V::Type{<:Potential}
    minlim::Vector{Float64}
    maxlim::Vector{Float64}
    C₀::Vector{Float64}
    Cπ::Vector{Float64}
    method::Symbol
    methodkws::Any
    kwargs::Any
end
function FitParamsPion(p::Params, V::Type{<:Potential}, C₀, Cpi::Vector{<:Real};
                       method=:LsqFit, methodkws=Dict(), minlim=nothing, maxlim=nothing, kwargs...)
    if isnothing(minlim)
        minlim = fill(-Inf, length(C₀) + length(Cpi))
    end
    if isnothing(maxlim)
        maxlim = fill(Inf, length(C₀) + length(Cpi))
    end
    FitParamsPion(p, V, minlim, maxlim, C₀, Cpi, method, methodkws, kwargs)
end

import Base.error

function fit(fparams::AbstractFitParams)
    solver = makesolver(fparams)
    res = solver()
    C = getsol(fparams, res)
    FitResult(fparams, res, C)
end

function getsol(fparams::AbstractFitParams, res)
    if fparams.method == :DE
        best_candidate(res)
    elseif fparams.method == :LsqFit
        res.param
    else
        Optim.minimizer(res)
    end
end

function makesolver(fparams::AbstractFitParams)
    V = fparams.V
    startpoint = fparams.params.start
    endpoint = fparams.params.stop
    length_ = fparams.params.length
    Λ = getΛ(fparams)
    _, reid = reidphase(fparams)
    erf = makeerrorfunction(V; startpoint, endpoint, length=length_, Λ, fact=reid)
    minlim = fparams.minlim
    maxlim = fparams.maxlim
    p₀ = getC₀(fparams)
    if length(minlim) == 1 && fparams.method ∉ Set([:LsqFit])
        () -> optimize(erf, minlim[1], maxlim[1])
    else
        if fparams.method in Set([:LBFGS, :GradientDescent])
            if fparams.method == :LBFGS
                inner = LBFGS(;fparams.methodkws...)
            elseif fparams.method == :GradientDescent
                inner = GradientDescent(;fparams.methodkws...)
            end
            () -> optimize(erf, minlim, maxlim, p₀, 
                           Fminbox(inner), Optim.Options(;fparams.kwargs...))
        elseif fparams.method == :Annealing
            () -> optimize(erf, minlim, maxlim, p₀, 
                           SAMIN(;fparams.methodkws...), Optim.Options(;fparams.kwargs...))
        elseif fparams.method == :ParticleSwarm
            () -> optimize(erf, minlim, maxlim, p₀,
                            ParticleSwarm(;fparams.methodkws...),
                            Optim.Options(;fparams.kwargs...))
        elseif fparams.method == :DE
            () -> bboptimize(erf; SearchRange = zip(minlim, maxlim) |> collect
                            ,fparams.kwargs...)
        elseif fparams.method == :LSOpt
            () -> LeastSquaresOptim.optimize(erf, p₀, LeastSquaresOptim.Dogleg();
                                             fparams.kwargs...)
        elseif fparams.method == :LsqFit
            X = computerange(fparams)
            E, reid = reidphase(fparams)
            function model(x, p)
                k = energytomomentum.(x)
                W = makeV(fparams, p)
                δ = phaseshift(k, mass, W, KMatrix(30))
            end
            () -> curve_fit(model, X, reid, p₀, lower=minlim, upper=maxlim)
        else
            throw(error(""))
        end
    end
end

computerange(p::AbstractFitParams) = computerange(p.params)
computerange(p::Params) = computerange(p.start, p.stop, p.length)

function makeV(f::FitParams, C)
    V = f.V(C...) |> x -> regularize(x, getΛ(f))
end
function makeV(f::FitParamsPion, C)
    n = length(f.C₀)
    V = f.V(C[1:n]...) + Pion(C[(n+1):end]...) |> x -> regularize(x, getΛ(f)) 
end
makeV(f::FitResult) = makeV(f.params, f.C)
getC₀(f::FitParams) = f.C₀
getC₀(f::FitParamsPion) = [f.C₀..., f.Cπ...]
    

function phase(V::Type{<:Potential}, coefficients, p::Params)
    V_ = V(coefficients...) |> x -> regularize(x, getΛ(p))
    E, δ = computephase(V_; startpoint=p.start,
    endpoint=p.stop, length=p.length)
end

function phase(V::Potential, p::Params)
    E, δ = computephase(V; startpoint=p.start,
    endpoint=p.stop, length=p.length)
end
function phase(coefficients, fparams::AbstractFitParams)
    V = makeV(fparams, coefficients)
    phase(V, fparams.params)
end

function phase(res::FitResult)
    phase(res.C, res.params)
end

function phase(res::FitResult, start, stop; length=350)
    V = makeV(res)
    E, δ = computephase(V; startpoint=start, endpoint=stop, length=length)
end

function reidphase(params::Params)
    E, δ = computephase(Reid(); startpoint=params.start, endpoint=params.stop,
        length=params.length) 
end

function reidphase(start, stop; length=350)
    E, δ = computephase(Reid(); startpoint=start, endpoint=stop,
        length=length) 
end
reidphase(p::AbstractFitParams) = reidphase(p.params)
reidphase(res::FitResult) = reidphase(res.params)


getΛ(p::Params) = p.Λ[1]
getΛ(p::AbstractFitParams) = getΛ(p.params)
getΛ(res::FitResult) = getΛ(res.params)

function error(res::FitResult, start, stop; length=350)
    _, δ = phase(res, start, stop; length)
    E, δreid = reidphase(start, stop; length)
    E, relerr(δreid, δ)
end

function error(res::FitResult)
    _, δ = phase(res)
    E, δreid = reidphase(res)
    E, relerr(δreid, δ)
end

