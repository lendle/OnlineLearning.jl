
export GLMModel, LogisticModel, LinearModel, QuantileModel
export grad_scratch!

##############################
## Used to specify GLM kind ##
## (link and family pair)   ##
##############################
abstract GLMModel
#Subtypes should define the loss, predict!, and grad_scratch! functions
# if the defaults are not appropriate.


function linpred!(pr::DenseVector{Float64},
                  coefs::Vector{Float64},
                  x::DSMat{Float64};
                  offset::Vector{Float64} = emptyvector(Float64))
        A_mul_B!(pr, x, coefs)
        if !isempty(offset)
            add!(pr, offset)
        end
        pr
end

#
# default predict! for GLMModel uses identity link
#
predict!(m::GLMModel,
        pr::DenseVector{Float64},
        coefs::Vector{Float64},
        x::DSMat{Float64};
        offset::Vector{Float64} = emptyvector(Float64)) = linpred!(pr, coefs, x; offset=offset)


# grad_scratch! calculates the gradient without allocating new memory
# for the residual vector.
# resid is scratch space and should have dims equal to size(y)
# Default is appropriate for GLMs with canonical link
function grad_scratch!(m::GLMModel,
                      gr::Vector{Float64},
                      coefs::Vector{Float64},
                      x::DSMat{Float64},
                      y::Vector{Float64},
                      resid::Vector{Float64};
                      offset::Vector{Float64} = emptyvector(Float64))
    predict!(m, resid, coefs, x; offset=offset)
    subtract!(resid, y) # resid = prediction - y
    alpha = 2.0/size(x, 1)
    At_mul_B!(alpha, x, resid, 0.0, gr)
    #BLAS.gemv!('T', alpha, x, resid, 0.0, gr) #grad \propto x'resid = -x'(y - pred)
    gr
end

grad!(m::GLMModel,
    gr::Vector{Float64},
    coefs::Vector{Float64},
    x::DSMat{Float64},
    y::Vector{Float64};
    offset::Vector{Float64} = emptyvector(Float64)) = grad_scratch!(m, gr, coefs, x, y, similar(y), offset=offset)

#########
## OLS ##
#########
type LinearModel <: GLMModel end

loss(m::LinearModel, pr::DenseVector{Float64}, y::Vector{Float64}) = meansqdiff(pr, y)


#########################
## Logistic regression ##
#########################
type LogisticModel <: GLMModel end

function predict!(m::LogisticModel,
                           pr::DenseVector{Float64},
                           coefs::Vector{Float64},
                           x::DSMat{Float64};
                           offset::Vector{Float64} = emptyvector(Float64))
    linpred!(pr, coefs, x; offset=offset)
    map1!(LogisticFun(), pr)
    pr
end


loss(m::LogisticModel, pr::DenseVector{Float64}, y::Vector{Float64}) = mean(NBLL(), pr, y)

#########################
## Quantile regression ##
#########################
type QuantileModel <: GLMModel
    tau :: Float64
end

QuantileModel() = QuantileModel(0.5)

function grad_scratch!(m::QuantileModel,
                               gr::Vector{Float64},
                               coefs::Vector{Float64},
                               x::DSMat{Float64},
                               y::Vector{Float64},
                               scratch::Vector{Float64};
                               offset::Vector{Float64} = emptyvector(Float64))
    predict!(m, scratch, coefs, x; offset=offset)
    subtract!(scratch, y)
    map1!(Qgrad(), scratch, m.tau)
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, scratch, 0.0, gr)
    gr
end

loss(m::QuantileModel, pr::DenseVector{Float64}, y::Vector{Float64}) = meanabsdiff(pr, y)
