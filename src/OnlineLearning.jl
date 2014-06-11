module OnlineLearning

using NumericExtensions, NumericFuns, Devectorize, ArrayViews

import NumericExtensions.evaluate

export Learner

abstract Learner
#should implement an update! and a predict! method
# update!{T<:FloatingPoint}(obj::Learner{T}, x::Matrix{T}, y::Vector{T})
#and possibly additional kw args

typealias DSMat{Tv, Ti <: Integer} Union(Matrix{Tv}, SparseMatrixCSC{Tv, Ti})

predict(obj::Learner, x::DSMat; kwargs...) =
	predict!(obj, Array(eltype(x), size(x,1)), x; kwargs...)


include("extra.jl")
include("sgd.jl")
include("glm.jl")
include("svm.jl")
include("utils.jl")

end # module
