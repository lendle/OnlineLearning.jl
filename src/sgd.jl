export AbstractSGD, SimpleSGD, AdaDelta, AdaGrad, AveragedSGD

abstract AbstractSGD
#should implement a update! method:
#update!(obj::AbstractSGD, weights::Vector{Float64}, gr::Vector{Float64})

#This function returns the vector of weights at which to evaluate the gradient
#It should be overridden by SGD implementations that need the gradient evaluated at
#values other than those stored in the Learner (AveragedSGD)
which_weights(obj::AbstractSGD, weights) = weights

type SimpleSGD <: AbstractSGD
    alpha1::Float64
    alpha2::Float64
    t::Int
    function SimpleSGD(alpha1::Float64, alpha2::Float64)
        alpha1 <= 0.0 && error("alpha1 should be positive")
        alpha2 < 0.0 && error("alpha2 should be non-negative")
        new(alpha1, alpha2, 0)
    end
end

function update!(obj::SimpleSGD, weights::Vector{Float64}, gr::Vector{Float64})
    obj.t += 1

    stepsize = - obj.alpha1 / (1.0 + obj.alpha1 * obj.alpha2 * obj.t)
    fma!(weights, gr, stepsize)
    weights
end

type AdaDelta <: AbstractSGD
    rho::Float64
    eps::Float64
    sqgr::Vector{Float64}
    squp::Vector{Float64}
    up::Vector{Float64}
    initialized::Bool
    function AdaDelta(rho::Float64, eps::Float64)
        (rho <= 0.0 || eps <= 0.0) && error("rho and epsilon should be positive")
        obj = new()
        obj.rho = rho
        obj.eps = eps
        obj.initialized = false
        obj
    end
end

Base.show(io::IO, obj::AdaDelta) = print(io, "AdaDelta(ρ=$(obj.rho), ε=$(obj.eps))")

function init!(obj::AdaDelta, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = zeros(weights)
    obj.squp = zeros(weights)
    obj.up = similar(weights)
    obj.initialized = true
end

function update!(obj::AdaDelta, weights::Vector{Float64}, gr::Vector{Float64})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] = obj.rho .* obj.sqgr + (1.0 - obj.rho) .* gr .* gr #line 4
    @devec obj.up[:] = - sqrt(obj.squp .+ obj.eps) ./ sqrt(obj.sqgr .+ obj.eps) .* gr #line 5
    @devec obj.squp[:] = obj.rho .* obj.squp + (1.0 - obj.rho) .* obj.up  .* obj.up #line 6
    add!(weights, obj.up) #line 7
end

type AdaGrad <: AbstractSGD
    eta::Float64
    sqgr::Vector{Float64}
    initialized::Bool
    function AdaGrad(eta::Float64)
        eta > 0.0 || error("eta should be positive")
        obj = new()
        obj.eta = eta
        obj.initialized = false
        obj
    end
end


Base.show(io::IO, obj::AdaGrad) = print(io, "AdaGrad(η=$(obj.eta))")

function init!(obj::AdaGrad, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = fill!(similar(weights), 1.0e-8)
    obj.initialized = true
end

function update!(obj::AdaGrad, weights::Vector{Float64}, gr::Vector{Float64})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] += gr .* gr
    @devec weights[:] -= obj.eta ./ sqrt(obj.sqgr) .* gr
    weights
end

type AveragedSGD <: AbstractSGD
    alpha1::Float64
    alpha2::Float64
    unaveraged_weights::Vector
    t0::Int
    t::Int
    initialized::Bool
    function AveragedSGD(alpha1::Float64, alpha2::Float64, t0::Int)
        alpha1 <= 0.0 && error("alpha1 should be positive")
        alpha2 < 0.0 && error("alpha2 should be non-negative")
        obj = new()
        obj.alpha1 = alpha1
        obj.alpha2 = alpha2
        obj.t0 = t0
        obj.t = 0
        obj.initialized = false
        obj
    end
end

Base.show(io::IO, obj::AveragedSGD) = print(io, "AveragedSGD(α1=$(obj.alpha1), α2=$(obj.alpha2), t0=$(obj.t0))")

function init!(obj::AveragedSGD, weights)
    obj.initialized && error("already initialized")
    obj.unaveraged_weights = zeros(weights)
    obj.initialized = true
end

which_weights(obj::AveragedSGD, weights) = obj.initialized? obj.unaveraged_weights: weights

function update!(obj::AveragedSGD, weights::Vector{Float64}, gr::Vector{Float64})
    obj.initialized || init!(obj, weights)
    obj.t += 1
    stepsize = - obj.alpha1 * (1.0 + obj.alpha1 * obj.alpha2 * obj.t)^-0.75
    fma!(obj.unaveraged_weights, gr, stepsize)
    mu = 1.0 / max(1.0, obj.t - obj.t0)
    @devec weights[:] += mu .* (obj.unaveraged_weights .- weights)
    weights
end
