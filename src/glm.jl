export GLMLearner, update!, grad!, predict!, loss, predict, GLMNetLearner, linpred, linpred!

include("glmmodel.jl")

abstract AbstractGLMLearner <: Learner


predict!(obj::AbstractGLMLearner, pr::DenseVector{Float64}, x::DSMat{Float64}; offset=emptyvector(Float64)) =
    predict!(obj.m, pr, obj.coefs, x, offset=offset)

predict(obj::AbstractGLMLearner, x::DSMat{Float64}; offset=emptyvector(Float64)) =
    predict!(obj, Array(Float64, size(x, 1)), x, offset=offset)

linpred!(obj::AbstractGLMLearner, pr::DenseVector{Float64}, x::DSMat{Float64}; offset=emptyvector(Float64)) =
    linpred!(pr, obj.coefs, x, offset=offset)

linpred(obj::AbstractGLMLearner, x::DSMat{Float64}; offset=emptyvector(Float64)) =
    linpred!(obj, Array(Float64, size(x, 1)), x, offset=offset)

loss(obj::AbstractGLMLearner, x::DSMat{Float64}, y::Vector{Float64}; offset=emptyvector(Float64)) =
    loss(obj.m, predict(obj, x, offset=offset), y)

################################
## GLM without regularization ##
################################
type GLMLearner <: AbstractGLMLearner
    m::GLMModel
    p::Int
    coefs::Vector{Float64}
    gr::Vector{Float64}
    optimizer::AbstractSGD
    initialized::Bool
    function GLMLearner(m::GLMModel, optimizer::AbstractSGD)
        obj = new()
        obj.m = m
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end


function Base.show(io::IO, obj::GLMLearner)
    print(io, "Model: ")
    show(io, obj.m)
    print(io, " Optimizer: ")
    show(io, obj.optimizer)
end

#Allocate obj.coefs and obj.gr on first call to update!
function init!(obj::GLMLearner, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(p)
    obj.gr = Array(Float64, p)
    obj.initialized = true
    obj
end

function update!(obj::GLMLearner, x::DSMat{Float64}, y::Vector{Float64}; offset=emptyvector(Float64), scratch=emptyvector(Float64))
    obj.initialized || init!(obj, size(x, 2))
    if length(scratch) == length(y)
        grad_scratch!(obj.m, obj.gr, which_weights(obj.optimizer, obj.coefs), x, y, scratch, offset=offset)
    else
        grad!(obj.m, obj.gr, which_weights(obj.optimizer, obj.coefs), x, y, offset=offset)
    end
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end

#############################
## GLM with regularization ##
#############################

type GLMNetLearner <: AbstractGLMLearner
    m::GLMModel
    lambda1::Float64
    lambda2::Float64
    p::Int
    coefs::Vector{Float64}
    gr::Vector{Float64}
    gr2::Vector{Float64}
    mu::Vector{Float64}
    nu::Vector{Float64}
    optimizer::AbstractSGD
    optimizer2::AbstractSGD
    initialized::Bool
    function GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1::Float64, lambda2::Float64)
        lambda1 < 0.0 || lambda2 < 0.0 && error("lambda1 and lambda2 should be non-negative")
        obj = new()
        obj.m = m
        obj.lambda1 = lambda1
        obj.lambda2 = lambda2
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end

GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0) = GLMNetLearner(m, optimizer, lambda1, lambda2)

function init!(obj::GLMNetLearner, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(p)
    obj.gr = Array(Float64, p)
    if obj.lambda1 > 0.0
        obj.mu = zeros(p)
        obj.nu = zeros(p)
        obj.gr2 = Array(Float64, p)
        obj.optimizer2 = deepcopy(obj.optimizer)
    end
    obj.initialized = true
    obj
end

function update!(obj::GLMNetLearner, x::DSMat{Float64}, y::Vector{Float64})
    obj.initialized || init!(obj, size(x,2))

    #calculate the gradient like usual
    grad!(obj.m, obj.gr,  which_weights(obj.optimizer, obj.coefs), x, y)
    if obj.lambda2 > 0.0
        #for l2 regularization, add l2 penalty
        fma!(obj.gr, obj.coefs, obj.lambda2) #gr = gr + lambda2 * coef
    end

    if obj.lambda1 > 0.0
        #for l1 regularization, calculate obj.gr - lambda1, stored in obj.gr2
        map!(Subtract(), obj.gr2, obj.lambda1, obj.gr)
        # then store obj.gr + lambda1 in obj.gr
        add!(obj.gr, obj.lambda1)
        update!(obj.optimizer, obj.mu, obj.gr) #update weights for both grads in mu, nu
        update!(obj.optimizer2, obj.nu, obj.gr2)
        map1!(MaxFun(), obj.mu, 0.0) #trunc temp weights at 0.0
        map1!(MaxFun(), obj.nu, 0.0)
        map!(Subtract(), obj.coefs, obj.mu, obj.nu) #calc final weights as mu - nu
    else
        #if no l1 regularization (so either only l2, or no reg at all)
        #just do standard update
        update!(obj.optimizer, obj.coefs, obj.gr)
    end
    obj
end
