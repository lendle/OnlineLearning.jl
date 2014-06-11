module Utils

using OnlineLearning

export fitlearner!

function fitlearner!(l::Learner, x, y, passes=1, batchsize=1, dumps=100; offset=:none)
    n = size(x, 1)

    batchesperpass, r = divrem(n, batchsize)
    if r!= 0
        warn("n is not divisible by batchsize, so some observations won't ever be used")
    end

    losses = Float64[]

    dumpinterval = dumps<=0? typemax(Int) : div(passes * batchesperpass, dumps)

    for pass in 1:passes
      for i in 1:batchesperpass
        r1= (i-1)*batchsize + 1
        r2 = i * batchsize
        if offset == :none
            update!(l, x[r1:r2, :], y[r1:r2])
        else
            update!(l, x[r1:r2, :], y[r1:r2], offset=offset[r1:r2])
        end

        if dumpinterval == 0 || i % dumpinterval == 0
            push!(losses, loss(l, x, y, offset=offset))
        end
      end
    end

    return (losses, dumpinterval)

end

end
