models = [LogisticModel(), LinearModel(), QuantileModel()]

#thest that loss and gradients agree
n=50
p=10
x = rand(n, p)
y = round(rand(n))
for model in models
    mylearner=  GLMLearner(model, SimpleSGD(1.0, 1.0))
    update!(mylearner, x, y)
    @test isa(predict(mylearner, x), Vector)

    analytic_grad = grad!(mylearner.m, mylearner.gr, OnlineLearning.which_weights(mylearner.optimizer, mylearner.coefs), x, y)
    numeric_grad = Calculus.gradient(coefs -> loss(mylearner.m, predict!(mylearner.m, similar(y), coefs, x), y), mylearner.coefs)

    @test_approx_eq_eps analytic_grad numeric_grad 1e-8
end

# myadagrad = GLMLearner(LogisticModel(), AdaGrad(1.0))
# update!(myadagrad, x, y)


# myglmnet = GLMNetLearner(LogisticModel(), AdaGrad(1.0), 0.1, 0.1)

# update!(myglmnet, x, y)

# update!(myglmnet, sprand(10, 10, 0.3), round(rand(10)))
