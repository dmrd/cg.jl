using cg
using Base.Test

# Test each operator
function test_gradients(out::cg.Node, shape=[5] ; filltype::Symbol=:range, iters=1, debug=false)
    G = cg.get_graph([out]);
    inputs = filter(x -> isa(x.op, cg.Placeholder), G.nodes)
    inputs = collect(inputs)
    gradients = cg.grad(out, inputs)

    session = cg.Session(out)
    for iter = 1:iters
        for (i,input) = enumerate(inputs)
            if filltype == :range
                # val[j] = i + j
                session.values[input] = reshape(collect(Float64, 1:(*(shape...))), shape...) + i
            elseif filltype == :rand
                session.values[input] = rand(shape...) - 0.5
            elseif filltype == :ones
                session.values[input] = ones(shape...)
            else
                @assert false && "invalid filltype"
            end
        end

        for (input,grad) = zip(inputs, gradients)
            numeric = cg.numeric_grad(session, out, input, 0.0000001)
            symbolic = cg.interpret(session, grad)
            if debug
                @show session.values[input]
                @show session.values[out]
                @show numeric
                @show symbolic
            end
            for i = 1:length(symbolic)
                @test_approx_eq_eps(symbolic[i], numeric[i], 0.0001)
            end
        end
    end
end

function test_gradients()
    i = () -> cg.placeholder([1]) # Shape doesn't matter yet
    @show @time test_gradients(sum(i()))

    # Scalar ops
    @show @time test_gradients(sum(i() + i()))
    @show @time test_gradients(sum(i() - i()))
    @show @time test_gradients(sum(i() * i()))
    @show @time test_gradients(sum(i() / i()))
    @show @time test_gradients(sum(i() ^ cg.constant(3.0)))
    @show @time test_gradients(sum(cg.constant(3.0) ^ i()))

    @show @time test_gradients(sum(-i()))
    @show @time test_gradients(sum(sign(i())))
    @show @time test_gradients(sum(sign(i())), filltype=:rand, iters=100)
    @show @time test_gradients(sum(exp(i())))
    @show @time test_gradients(sum(log(i())))
    @show @time test_gradients(sum(sin(i())))
    @show @time test_gradients(sum(cos(i())))
    @show @time test_gradients(sum(abs(i())))

    @show @time test_gradients(sum(max(i(), i())))
    # Leave this and min() one out for now - what's the
    # intended behavior when equal?
    #@show @time test_gradients(sum(max(i(), i())), filltype=:ones)
    @show @time test_gradients(sum(max(i(), i())), filltype=:rand, iters=100)

    @show @time test_gradients(sum(min(i(), i())))
    #@show @time test_gradients(sum(min(i(), i())), filltype=:ones)
    @show @time test_gradients(sum(min(i(), i())), filltype=:rand, iters=100)

    @show @time test_gradients(sum(cg.sigmoid(i())))

    # Other ops
    @show @time test_gradients(sum(maximum(i())))
    @show @time test_gradients(sum(maximum(i())), filltype=:ones)
    @show @time test_gradients(sum(maximum(i())), filltype=:rand, iters=100)
    @show @time test_gradients(sum(maximum(i(), cg.constant(1))), [10,15], filltype=:rand, iters=100)
    @show @time test_gradients(sum(maximum(i(), cg.constant(2))), [10,15], filltype=:rand, iters=100)

    # Mat mul
    @show @time test_gradients(cg.dot(cg.t(i()), i()))
end

function test_sgd_basics()
    # Test basic optimization - minimize sum((b - a)^2)
    target = rand(10)
    a = cg.constant(target, "a")
    b = cg.variable(cg.constant(zeros(10)), "b") # Parameter to optimize
    c = b - a
    d = c * c
    e = sum(d)
    values = Dict{cg.Node, cg.TensorValue}()
    optimizer = cg.sgd_optimizer(e, [b], cg.constant(0.001, "step_size"))
    #cg.render(cg.get_graph([b]), "graph.png")
    session = cg.Session(optimizer)
    for i = 1:10000
        cg.interpret(session, optimizer)
        if session.values[e][1] <= 0.001
            break
        end
    end
    @test_approx_eq_eps target session.values[b] 0.1
end

# Test that sum gradients work properly
function test_sum()
    a = cg.placeholder([1])
    b = cg.variable(cg.randn(cg.constant([1,5])))
    c = cg.sum(a, cg.constant(1))
    d = c * b
    e = sum(d)
    test_gradients(e, [3, 5])
end


test_gradients()
test_sgd_basics()
test_sum()
