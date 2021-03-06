# OnlineLearning

[![Build Status](https://img.shields.io/travis/lendle/OnlineLearning.jl.svg?style=flat)](https://travis-ci.org/lendle/OnlineLearning.jl)
[![Coverage Status](https://img.shields.io/coveralls/lendle/OnlineLearning.jl.svg?style=flat)](https://coveralls.io/r/lendle/OnlineLearning.jl)

An implementation of online mini-batch learning for prediction in julia.

## Learners

A `Learner` is fit by repeated calls to `update!(l::Learner, x::DSMat{Float64}, y::Vector{Float64})` on mini-batches `(x, y)` of a dataset.
Updating a learner incrementally optimizes some loss function.
The loss function depends on the implementation of concrete subtypes of `Learner`.
The actual optimization routine is implemented by an `AbstractSGD` object.

Values of the outcome are predicted with `predict(l::Learner, x::DSMat{Float64})`. The `predict!(obj::Learner, pr::Vector{Float64}, x::DSMat{Float64})` method calculates predictions in place.

Features (`x`) can be either a dense or sparse matrix. (`DSMat{T}` is an alias for `DenseMatrix{T}` or `SparseMatrixCSC{T, Ti <: Integer}`)

### Available learners

* `GLMLearner(m::GLMModel, optimizer::AbstractSGD)` - GLMs without regularization.
* `GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0)` - GLMs with l_1 and l_2 regularization.
* `SVMLearner` - support vector machine, not fully implemented

The type of GLM is specified by `GLMModel`. Choices are:

* `LinearModel()` for least squares
* `LogisticModel()` for logistic regression
* `QuantileModel(tau=0.5)` for `tau`-quantile regression.

## Optimization

All of the learners require an optimizer of some sort.
Currently, stochastic gradient descent type methods are provided by the `AbstractSGD` type.

An `AbstractSGD` implements an `update!(obj::AbstractSGD{Float64}, weights::Vector{Float64}, gr::Vector{Float64})` method.
This takes the current value of the weight(coefficient) vector and gradient and updates the weight vector in place.
The `AbstractSGD` instance stores tuning parameters and step information, and may have additional storage additional storage for if necessary.

### Available optimizers:

* `SimpleSGD(alpha1::Float64, alpha2::Float64)` - Standard SGD where step size is `alpha1/(1.0 + alpha1 * alpha2 * t)`.
* `AdaDelta(rho::Float64, eps::Float64)` - Implementation of Algorithm 1 [here](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf).
* `AdaGrad(eta::Float64)` Stepsize is for weight `j` `eta /[sqrt(sum of grad_j^2 up to t) + 1.0e-8]`. [Paper](http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf).
* `AveragedSGD(alpha1::Float64, alpha2::Float64, t0::Int)` - Described in section 5.3 [here](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) with step size `alpha1/(1.0 + alpha1 * alpha2 * t)^(3/4)`

## Notes

This is a work in progress. Most testing has been in simulations and not with real data. `GLMLearner` and `GLMNetLearner` with l_2 regularization seem to work pretty well. `GLMNetLearner` with l_1 regularization has not been thoroughly tested. Statistical performance tends to be pretty senstive to choice of optimizer and tuning parameters.



### TODO

* Everything is implemented in terms of Float64. Should allow for Float32 as well.
* Finish the SVM implementation, perhaps add Pegasos implementation
* Automatic transformations of features
* More useful interfaces/DataFrames interface
* More checking of data
* Automatic bounding for predictions
* Remove GLMLearner in favor of GLMNetLearner
* Better docs
* Eliminate extra memory allocation
