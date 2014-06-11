#Check AdaDelta works
mylearner=  GLMLearner(LogisticModel(), AdaDelta(1.0, 1.0))

x = rand(10, 10)
y = round(rand(10))

update!(mylearner, x, y)

predict(mylearner, x)


myadagrad = GLMLearner(LogisticModel(), AdaGrad(1.0))
update!(myadagrad, x, y)


myglmnet = GLMNetLearner(LogisticModel(), AdaGrad(1.0), 0.1, 0.1)

update!(myglmnet, x, y)
