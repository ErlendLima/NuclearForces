import PyPlot
const plt = PyPlot
using LaTeXStrings
using PyCall
const matplotlib = pyimport("matplotlib")
using Formatting: format
#include("./setup.jl")


#### SETUPFILE

using Scattering
using Statistics
using Optim
using LineSearches
using BlackBoxOptim
using LsqFit

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
        if arr[i]*arr[i-1] < 0 && arr[i] > 0.5
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
    end
    factcot = @. k*cot(fact |> deg2rad)
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
function FitParams(params, V::Type{<:Potential}; C₀, minlim=nothing, maxlim=nothing, kwargs...) 
    if isnothing(minlim)
        minlim = fill(-Inf, length(C₀))
    end
    if isnothing(maxlim)
        maxlim = fill(Inf, length(C₀))
    end
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

struct FitParamsPurePion <: AbstractFitParams
    params::Params
    V::Type{<:Potential}
    minlim::Vector{Float64}
    maxlim::Vector{Float64}
    C₀::Vector{Float64}
    method::Symbol
    methodkws::Any
    kwargs::Any
end
function FitParamsPurePion(p::Params, Cpi;
                       method=:LsqFit, methodkws=Dict(), minlim=nothing, maxlim=nothing, kwargs...)
    if isnothing(minlim)
        minlim = fill(-Inf, 1)
    end
    if isnothing(maxlim)
        maxlim = fill(Inf, 1)
    end
    FitParamsPurePion(p, Pion, minlim, maxlim, Cpi, method, methodkws, kwargs)
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
function makeV(f::FitParamsPurePion, C)
    V = Pion(C...) |> x -> regularize(x, getΛ(f))
end

makeV(f::FitResult) = makeV(f.params, f.C)
getC₀(f::FitParams) = f.C₀
getC₀(f::FitParamsPion) = [f.C₀..., f.Cπ...]
getC₀(f::FitParamsPurePion) = f.C₀

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


function getlim(lim)
    if typeof(lim) <: Real
        x -> lim
    elseif !isempty(methods(lim))
        lim
    end
end

function intp(p1, p2; atleast=nothing, atmost=nothing)
    if p1[1] > p2[1]
        tmp = p1
        p1 = p2
        p2 = p1
    end
    a = (p2[2]-p1[2])/(p2[1]-p1[1])
    b = p2[2] - a*p2[1]

    if !isnothing(atleast)
        x -> max(atleast, 197a*x+b)
    elseif !isnothing(atmost)
        x -> min(atmost, 197a*x+b)
    else
        x -> 197a*x+b
    end
end

function measurelambda(V, coeffs::AbstractVector, Cpi::AbstractVector,
                       lambdas::AbstractVector; endpoint=1e-1, minlim=nothing, maxlim=nothing)
    results = FitResult[]
    # A trick to adjust the limits of Vpi
    adjustmin = !isnothing(minlim)
    adjustmax = !isnothing(maxlim)
    _minlim = if adjustmin minlim .|> getlim else minlim end
    _maxlim = if adjustmax maxlim .|> getlim else maxlim end

    for Λ in lambdas |> tqdm
        minlim = if adjustmin
            [f(Λ) for f in _minlim] else minlim end
        maxlim = if adjustmax
            [f(Λ) for f in _maxlim] else maxlim end
        @show minlim
        @show maxlim
           # minlim[end] = mi(-0.08, -1.0+1.0/500*197Λ)
           # maxlim[end] = max(-0.01, -0.01/500*197Λ)
        p = Params(1e-3, endpoint, 10, [Λ])
        fp = FitParamsPion(p, V, coeffs, Cpi, method=:LsqFit,
                           ;minlim, maxlim)
        res = fit(fp)
        push!(results, res)
    end
    results
end

function measurelambda(V, coeffs::AbstractVector,
                       lambdas::AbstractVector; endpoint=1e-1, minlim=nothing, maxlim=nothing)
    results = FitResult[]
    adjustmin = !isnothing(minlim)
    adjustmax = !isnothing(maxlim)
    _minlim = if adjustmin minlim .|> getlim else minlim end
    _maxlim = if adjustmax maxlim .|> getlim else maxlim end
    for Λ in lambdas |> tqdm
        minlim = if adjustmin
            [f(Λ) for f in _minlim] else minlim end
        maxlim = if adjustmax
            [f(Λ) for f in _maxlim] else maxlim end
        @show minlim
        @show maxlim
        p = Params(1e-3, endpoint, 10, [Λ])
        fp = FitParams(p, V; C₀=coeffs, method=:LsqFit, minlim, maxlim)
        res = fit(fp)
        push!(results, res)
    end
    results
end

function measurelambda(V::Type{Pion}, coeffs::AbstractVector,
                       lambdas::AbstractVector; endpoint=3, minlim=nothing, maxlim=nothing)
    results = FitResult[]
    for Λ in lambdas |> tqdm
        p = Params(1e-3, endpoint, 10, [Λ])
        fp = FitParamsPurePion(p, coeffs, method=:LsqFit,
                       ;minlim, maxlim)
        res = fit(fp)
        push!(results, res)
    end
    results
end
#### END SETUPFIL

function savefig(fig, name; dpi=196, kwargs...)
    path = joinpath("../latex/Figures/", name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", kwargs...)
end


function updaterc()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    basesize = 12
    rcParams["backend"] = "ps"
    #rcParams["font.family"] = 
    rcParams["font.size"] = basesize
    rcParams["axes.labelsize"] = basesize
    rcParams["legend.fontsize"] = basesize
    rcParams["xtick.labelsize"] = basesize-2
    rcParams["ytick.labelsize"] = basesize-2
    rcParams["text.usetex"] = true
    rcParams["text.color"] = "252525"
    rcParams["xtick.major.size"] = 3
    rcParams["ytick.major.size"] = 3
    rcParams["xtick.minor.size"] = 1
    rcParams["hatch.linewidth"] = 0.1  # previous pdf hatch linewidth 
end
plt.matplotlib.style.use("rapport")
updaterc()

function cmap(name, number)
    cm = plt.cm.get_cmap(name)
    Ncolors = min(cm.N, number)
    mapcolors = [cm(round(Int, x*cm.N/Ncolors)) for x in 1:Ncolors]
end
++(a, b) = append!(collect(a), collect(b))

function newfig(;mul=1, height=nothing, kwargs...)
    fig_width_pt = 467.42 # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    if height != nothing
        fig_height *= height 
    end
    nrows = get(kwargs, :nrows, 1)
    fig_size = mul*[fig_width,nrows*fig_height] 
    plt.subplots(figsize=(fig_size); kwargs...)
end

function pi_ticks(ax;base=0.5)
    ticker = matplotlib.ticker.FormatStrFormatter(raw"$%g \: \pi$")
    locator = matplotlib.ticker.MultipleLocator(base=base)
    ax.set_major_formatter(ticker)
    ax.set_major_locator(locator)
end

function twinscale(ax; func=x -> x*197)
    ax2 = ax.twiny()
    function convert_ax(ax_f)
        """
        Update second axis according with first axis.
        """
        y1, y2 = ax_f.get_xlim()
        @show y1, y2, func(y1), func(y2)
        ax2.set_xlim(func(y1), func(y2))
        ax2.figure.canvas.draw()
    end
    ax.callbacks.connect("xlim_changed", convert_ax)
    return ax2
end

index(xs, x) = argmin(@. abs(xs - x))

label(fp::FitParams) = label(fp.V)
label(fp::FitParamsPion) = label(fp.V) * L"+\pi"
label(fp::FitParamsPurePion) = L"\pi"
label(f::FitResult) = label(f.params)
label(::Type{LO}) = "LO"
label(::Type{NLO}) = "NLO"
label(::Type{NNLO}) = "NNLO"
label(::Type{Reid}) = "Reid"
label(::Type{Pion}) = L"\pi"
getlabel = label

function plotphase(res::Vector{FitResult}, start, stop; length=350, ax=nothing,
    legend=true)
    if isnothing(ax)
        fig, ax = newfig()
    end
    E, δReid = reidphase(start, stop; length)
    
    ax.plot(E, δReid, label="Reid", linestyle="--", c="k", zorder=10)
    for r in res
        E, δ = phase(r, start, stop; length)
        ax.plot(E, δ, label=label(r))
    end
    ax.set_xlim(start, stop)
    ax.axhline(y=0, c="k", linewidth=1, alpha=0.5)
    ax.set_xlabel(L"$T_{\mathrm{Lab}}$ [MeV]")
    ax.set_ylabel(L"$\delta_0$ [deg]")
    ax.set_xscale("log")
    legend && ax.legend()
    plotfitregion(res[1], ax)
    ax
end

function plotphase(res::FitResult, start, stop; length=350, ax=nothing, label=nothing)
   if isnothing(ax)
        fig, ax = newfig()
    end
    if isnothing(label)
        label = getlabel(res)
    end
    E, d = phase(res, start, stop; length)
    ax.plot(E, d, label=label)
    ax
end
function ploterror(res::FitResult, start, stop; length=350, ax=nothing, label=nothing)
   if isnothing(ax)
        fig, ax = newfig()
    end
    if isnothing(label)
        label = getlabel(res)
    end
    E, d = error(res, start, stop)
    ax.plot(E, d, label=label)
    ax
end
function ploterror(res::Vector{FitResult}, start, stop; length=350, ax=nothing,
    legend=true, region=true)
    if isnothing(ax)
        fig, ax = newfig()
    end
   
    for r in res
        E, Δδ = error(r, start, stop)
        ax.plot(E, Δδ, label=label(r))
    end
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$T_{\mathrm{Lab}}$ [MeV]")
    ax.set_ylabel("Relative error")
    legend && ax.legend()
    region && plotfitregion(res[1], ax)
    ax
end

function plotdiff(res::Vector{FitResult}, start, stop; length=350, ax=nothing,
    legend=true)
    if isnothing(ax)
        fig, ax = newfig()
    end
   
    E, δReid = reidphase(start, stop; length)
    for r in res
        E, δ = phase(r, start, stop; length)
        diff = δReid .- δ
        ax.plot(E, diff, label=label(r))
    end
    ax.axhline(y=0, c="k", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$T_{\mathrm{Lab}}$ [MeV]")
    ax.set_ylabel("Difference in phase shift [deg]")
    legend && ax.legend()
    plotfitregion(res[1], ax)
    ax
end


function plotfitregion(res::FitResult, ax)
   E = computerange(res.params.params.start, res.params.params.stop,
        res.params.params.length)
    m = []
    for line in ax.lines
        data = line._yorig
        push!(m, minimum(data))
    end
    offset = minimum(m)
    ax.plot(E, offset * ones(length(E)), "|", color="grey")
end

function computerrors(res::Vector{FitResult}; start=1e-3, stop=1e2)
    V = Vector{Float64}[]
    errp = Vector{Float64}[]
    errors = Float64[]
    E, reid = reidphase(start, stop, length=350)
    for fit in res
        _, δ = phase(fit, start, stop, length=350)
        push!(V, δ)
        err = relerr(reid, δ)
        push!(errp, err)
        err = error(fit)[2] |> sum
        push!(errors, err)
    end
    E, reid, V, errp, errors
end

function plotfit(iter, E, reid, phases, errp)
    fig, (ax1, ax2) = newfig(nrows=2, constrained_layout=true, sharex=true)
    cmap_ = "turbo"
    norm = plt.matplotlib.colors.Normalize(vmin=minimum(iter), vmax=maximum(iter))
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm)
    cm = cmap(sm.cmap, length(phases)) #cmap(sm, length(Vs))
    ax1.plot(E, reid, "k-")
    ax1.axhline(y=0, c="gray", linewidth=0.5)
    for (i, (point, phase)) in zip(iter, phases) |> enumerate
        ax1.plot(E, phase, color=cm[i], alpha=1)
        ax2.plot(E, errp[i], color=cm[i])
    end

    ax1.set_xlim(1e-3, 1e2)
    ax1.set_ylim(0, 70)

    cbar = fig.colorbar(sm, ax=(ax1, ax2), shrink=0.6)#, location="right")
    cbar.ax.set_ylabel(L"$\Lambda$ [fm-1]")
    ax1.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel(L"$\delta$ [deg]")
    ax2.set_ylabel("Relative difference")
    ax1.legend()
    #ax1.set_xlim(1e-2, 20)
    ax2.set_xlabel(L"$T_{\mathrm{lab}}$ [MeV]")
        [ax1, ax2], cbar
end

function getci(res, i)
    c = [fit.C[i] for fit in res]
    low, high = zip([confidence_interval(fit.res, 0.05)[i] for fit in res]...)
    c, (low, high)
end

function plotci(res, i, ax, x; bounds=true)
    c, (low, high) = getci(res, i)
    if bounds
        ax.fill_between(x, low, high, alpha=0.3)
    end
    ax.plot(x, c)
end

function printci(res, i, ibest)
    c = res[ibest].C[i]
    low, high = confidence_interval(res[ibest].res, 0.05)[i]
    @show low, high
    ch = high - c
    cl = c - low
    pl = floor(Int, log10(cl))
    ph = floor(Int, log10(ch))
    precision = min(pl, ph) |> abs
    pl = abs(pl)
    ph = abs(ph)
    s = "{{{:.$(precision)f}}}^{{{:.$(ph)f}}}_{{{:.$(pl)f}}}"
    s2 = "{:.$(precision)f} +- {:.$(precision)f}"
    s2 = raw"\num{"*format(s2, c, cl)*"}"
    s = format(s,c,ch,cl)
    #println(LaTeXString(raw"$"*s*raw"$"))
    s, s2
end

struct lambdaresult
    res
    E
    reid
    V
    errp
    error
    lambdas
    bestlambda
    lmin
end
label(r::lambdaresult) = label(r.res[1])

function fitlambdas(V::Type{<:Potential}, coeffs, ;start, stop, length=10, vargs...)
    lambdas = range(start, stop, length=length)/197 |> collect
    res = measurelambda(V, coeffs, lambdas; vargs...);
    E, reid, V, errp, error = computerrors(res);
    lmin = argmin(errp)
    lambdas *= 197
    bestlambda = lambdas[lmin]
    println("Best Λ: $bestlambda")
    return lambdaresult(res, E, reid, V, errp, error, lambdas, bestlambda, lmin)
end


function fitlambdas(V::Type{<:Potential}, coeffs, cpi;start, stop, length=10, vargs...)
    lambdas = range(start, stop, length=length)/197 |> collect
    res = measurelambda(V, coeffs, cpi, lambdas; vargs...);
    E, reid, V, errp, error = computerrors(res);
    lmin = argmin(errp)
    lambdas *= 197
    bestlambda = lambdas[lmin]
    println("Best Λ: $bestlambda")

    return lambdaresult(res, E, reid, V, errp, error, lambdas, bestlambda, lmin)
    #return lambdas, E, reid, V, errp, error, lmin
end

plotlambdaerror(res::lambdaresult) = plotlambdaerror(res.E, res.lambdas, res.lmin, res.errp, res.error)
function plotlambdaerror(E, lambdas, lmin, errp, error)
    fig, (ax1, ax2) = newfig(nrows=2, constrained_layout=true)
    ax1.plot(lambdas, error, label="")
    ax1.axvline(lambdas[lmin], color="C1", label="Minimum")
    ax1.set_ylabel("Total relative error")
    ax1.set_xlabel(L"$\Lambda$ [MeV]")
    ax1.legend()
    ax1.set_yscale("log")

    norm = plt.matplotlib.colors.Normalize(vmin=minimum(lambdas), vmax=maximum(lambdas))
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    cm = cmap(sm.cmap, length(lambdas)) #cmap(sm, length(Vs))

    for (i, err) in errp |> enumerate
        ax2.plot(E, err, color=cm[i])
    end
    ax2.plot(E, errp[lmin], "k--")
    # for (i, err) in errp |> enumerate
    #     ax2.plot(E, cumsum(err), color=cm[i])
    # end
    # ax2.plot(E, cumsum(errp[lmin]), "k--")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylabel("Relative error")
    ax2.set_xlabel(L"$T_{\mathrm{lab}}$ [MeV]")

    cbar = fig.colorbar(sm, ax=ax2)#, location="right")
    cbar.ax.set_ylabel(L"$\Lambda$ [MeV]")
    cbarxlim = cbar.ax.get_xlim()
    cbar.ax.plot(cbarxlim, [lambdas[lmin], lambdas[lmin]], "k-", lw=2)

    mpi = 139.6
    cbar.ax.plot(cbarxlim, [mpi, mpi], "w", lw=2)
    return fig
end


function getci(res::lambdaresult, i)
    c = [fit.C[i] for fit in res.res]
    low, high = zip([confidence_interval(fit.res, 0.05)[i] for fit in res.res]...)
    c, (low, high)
end

function plotci(res::lambdaresult; ax=nothing, bounds=true)
    N = length(res.res[1].C)
    @show N
    if isnothing(ax)
        fig, ax = newfig(nrows=N, sharex=true, constrained_layout=true,
                         height=2/N)
        if N == 1
            ax = [ax]
        end
    end
    for i in 1:N
        plotci(res, i, ax[i]; bounds)
    end
    ax
end

function plotci(res::lambdaresult, i::Integer, ax; bounds=true, limits=false)
    c, (low, high) = getci(res, i)
    if bounds
        ax.fill_between(res.lambdas, low, high, alpha=0.3)
    end
    ax.plot(res.lambdas, c)
    if limits
        ax.set_ylim(0.8*c[res.lmin], 1.2*c[res.lmin])
    end
    ax
end

function printci(res::lambdaresult)
    N = length(res.res[1].C)
    for i in 1:N
        printci(res, i) |> println
    end
end
printci(res::lambdaresult, i::Integer) = printci(res.res, i, res.lmin)

plotlambdafit(res::lambdaresult) = plotlambdafit(res.E, res.reid, res.lambdas, res.V, res.errp, res.lmin)

function plotlambdafit(E, reid, lambda, Vs, errps, emin)
    fig, (ax1, ax2) = newfig(nrows=2, constrained_layout=true, sharex=true)
    cmap_ = "turbo"
    norm = plt.matplotlib.colors.Normalize(vmin=minimum(lambda), vmax=maximum(lambda))
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm)
    cm = cmap(sm.cmap, length(lambda)) #cmap(sm, length(Vs))
    alphas = map(x -> if findjump(x) 0.1 else 1.0 end, Vs)

    for (i, v) in Vs |> enumerate
        ax1.plot(E, v, color=cm[i], alpha=alphas[i])
    end
    ax1.plot(E, reid, "k-", lw=1, label="Reid")
    ax1.plot(E, Vs[emin], "k--", lw=1, label="Best fit")
    ax1.legend()

    for (i, err) in errps |> enumerate
        ax2.plot(E, err, color=cm[i], alpha=alphas[i])
    end
    ax2.plot(E, errps[emin], "k--")
    ax1.set_ylim(0, 70)
    ax1.set_xlim(1e-3, 100)
    ax1.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel(L"$\delta$ [deg]")
    ax2.set_ylabel("Relative error")
    ax2.set_xlabel(L"$T_{\mathrm{lab}}$ [MeV]")
    cbar = fig.colorbar(sm, ax=(ax1, ax2), shrink=0.6)#, location="right")
    cbar.ax.set_ylabel(L"$\Lambda$ [MeV]")
    cbarxlim = cbar.ax.get_xlim()

    mpi = 139.6
    lbest = lambda[emin]
    cbar.ax.plot(cbarxlim, [mpi, mpi], "w", lw=2)
    cbar.ax.plot(cbarxlim, [lbest, lbest], "k-", lw=2)
    return fig
end
