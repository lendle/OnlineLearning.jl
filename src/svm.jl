export SVMLearner

type SVMLearner <: Learner
    lambda::Float64
    p::Int
    coefs::Vector{Float64}
    gr::Vector{Float64}
    optimizer::AbstractSGD
    initialized::Bool
    function SVMLearner(optimizer::AbstractSGD, lambda::Float64)
        lambda >= 0.0 || error("lambda should be non-negative")
        obj = new()
        obj.optimizer = optimizer
        obj.lambda = lambda
        obj.initialized = false
        obj
    end
end

SVMLearner(optimizer::AbstractSGD, lambda::Float64) = SVMLearner{Float64}(optimizer, lambda)

function Base.show(io::IO, obj::SVMLearner)
    print(io, "SVMLearner lambda: $(obj.lambda)")
    print(io, "\nOptimizer: ")
    show(io, obj.optimizer)
    if obj.initialized
        print(io, "\nCoefficients: ")
        show(io, obj.coefs)
    end
end

function init!(obj::SVMLearner, p)
    obj.initialized && error("already initialized")
    obj.coefs = zeros(p)
    obj.gr = Array(Float64, p)
    obj.initialized = true
end

function grad!(obj::SVMLearner, x, y)
   error()
end

function update!(obj::SVMLearner, x::DSMat{Float64}, y::Vector{Float64})
    obj.initialized || init!(obj, size(x, 2))
    grad!(obj, x, y)
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end
