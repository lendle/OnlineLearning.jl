#negative Bernoulli log likelihood
type NBLL <: Functor{2} end
evaluate(::NBLL, pr, y) = y == one(y)? -log(pr):
                          y == zero(y)? -log(one(y) - pr):
                          -(y * log(pr) + (one(y) - y) * log(one(pr) - pr))

# Gradient helper for quantile regression
type Qgrad <: Functor{2} end
evaluate(::Qgrad, r, tau) = r == zero(r)? zero(r) :
                                 r < zero(r)? oftype(r, -tau):
                                 oftype(r, 1.0-tau)

emptyvector{T}(::Type{T}) = Array(T, 0)

Base.At_mul_B!{T<:Base.LinAlg.BlasFloat}(α::Number,A::StridedMatrix{T},x::StridedVector{T},β::Number,y::StridedVector{T}) =
  BLAS.gemv!('T', oftype(T, α), A, x, oftype(T, β), y)
