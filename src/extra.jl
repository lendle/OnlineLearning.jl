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


function projectsimplex{T <: Real}(v::Array{T, 1}, z::T=one(T))
  n=length(v)
  µ = sort(v, rev=true)
  #finding ρ could be improved to avoid so much temp memory allocation
  ρ = maximum((1:n)[µ - (cumsum(µ) .- z) ./ (1:n) .>0])
  θ = (sum(µ[1:ρ]) - z)/ρ
  max(v .- θ, 0)
end


Base.At_mul_B!{T<:Base.LinAlg.BlasFloat}(α::Number,A::StridedMatrix{T},x::StridedVector{T},β::Number,y::StridedVector{T}) =
  BLAS.gemv!('T', oftype(T, α), A, x, oftype(T, β), y)
