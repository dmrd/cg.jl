using cg
using Base.Test
using Base

show_test(test::AbstractString) = println("== Running tests $test ==")
i = () -> cg.placeholder([1]) # Shape doesn't matter yet
c = cg.constant

function genarg(shape::AbstractArray, filltype::Symbol ; offset = 0, rand_offset = -0.5)
    if filltype == :range
        # val[j] = i + j
        return reshape(collect(Float64, 1:(*(shape...))), shape...) + offset
    elseif filltype == :rand
        return rand(shape...) + rand_offset
    elseif filltype == :ones
        return ones(shape...)
    else
        @assert false && "invalid filltype"
    end
end

function get_inputs(n::cg.Node)
    G = cg.get_graph([n])
    inputs = filter(x -> isa(x.op, cg.Placeholder), G.nodes)
    collect(inputs)
end


function test_output(out::cg.Node, truth::Function, args::Tuple{cg.Node, Array{Float64, 2}}...)
    # Float64/2 should be TensorValue once type system updated
    session = cg.Session(out)
    for (node, value) in args
        session.values[node] = value
    end
    graphresult = cg.interpret(session, out)
    truthresult = truth([x[2] for x in args]...)
    @test_approx_eq_eps(graphresult, truthresult, 0.001)
end

# Test each operator
function test_gradients(out::cg.Node, shape=[5] ; filltype::Symbol=:range, iters=1, debug=false)
    inputs = get_inputs(out)
    gradients = cg.grad(out, inputs)

    session = cg.Session(out)
    for iter = 1:iters
        for (i,input) = enumerate(inputs)
            session.values[input] = genarg(shape, filltype, offset = i)
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

i = () -> cg.placeholder([1]) # Shape doesn't matter yet
function test_scalar_gradients()
    show_test("scalar")
    @show test_gradients(sum(i()))

    # Scalar ops
    @show test_gradients(sum(i() + i()))
    @show test_gradients(sum(i() - i()))
    @show test_gradients(sum(i() * i()))
    @show test_gradients(sum(i() / i()))
    @show test_gradients(sum(i() ^ cg.constant(3.0)))
    @show test_gradients(sum(cg.constant(3.0) ^ i()))

    @show test_gradients(sum(-i()))
    @show test_gradients(sum(sign(i())))
    @show test_gradients(sum(sign(i())), filltype=:rand, iters=100)
    @show test_gradients(sum(exp(i())))
    @show test_gradients(sum(log(i())))
    @show test_gradients(sum(sin(i())))
    @show test_gradients(sum(cos(i())))
    @show test_gradients(sum(abs(i())))

    @show test_gradients(sum(max(i(), i())))
    # Leave this and min() one out for now - what's the
    # intended behavior when equal?
    #@show test_gradients(sum(max(i(), i())), filltype=:ones)
    @show test_gradients(sum(max(i(), i())), filltype=:rand, iters=100)

    @show test_gradients(sum(min(i(), i())))
    #@show test_gradients(sum(min(i(), i())), filltype=:ones)
    @show test_gradients(sum(min(i(), i())), filltype=:rand, iters=100)

    @show test_gradients(sum(cg.sigmoid(i())))
end

function test_shape_gradients()
    show_test("shape")
    # Shape related ops
    shape = [10,15]
    ops = [cg.maximum, cg.sum]
    for op in ops
        @show op
        for graph in [sum(op(i())),
                      sum(op(i(), cg.constant(1))), sum(op(i(), cg.constant(2)))]
            for filltype = [:ones, :rand, :range]
                @show (op, graph, filltype)
                #test_gradients(graph, shape, filltype=filltype, iters=100)
            end
        end
    end
end

function test_get_and_set_gradients()
    c = cg.constant
    show_test("get_and_set")
    @show test_gradients(cg.getindex(i(), c(4)))
    @show test_gradients(cg.getindex(cg.setindex(i(), c(3), c(1000)), c(3)))
    @show test_gradients(cg.getindex(cg.setindex(i(), c(3), c(1000)), c(2)))
end

function test_other_gradients()
    show_test("other gradients")
    # Mat mul
    @show test_gradients(cg.dot(cg.t(i()), i()))
end

function test_broadcast()
    i1 = i()
    add = cg.broadcastop(cg.Add(), c([10.0, 10.0]), i1)
    #@show test_output(add, (x) -> [10.0, 10.0] .+ x, (i1, [1.0]'))
    @show test_gradients(cg.sum(add), [1], debug=true, filltype=:ones)
end

function test_nn()
    show_test("nn")

    tests = [cg.sum(cg.softmax(i())),
             cg.sum(cg.crossentropy(i(), cg.softmax(i())))]
    for filltype in [:ones, :rand]
        si1 = i()
        softmax = cg.softmax(si1)
        ci1 = i()
        ci2 = i()
        crossentropy = cg.crossentropy(ci1, ci2)
        sci1 = i()
        sci2 = i()
        softmax_crossentropy = cg.crossentropy(sci1, cg.softmax(sci2))

        softmax_t(x) = exp(x) ./ sum(exp(x), 1)
        crossentropy_t(p, q) = -sum(p .* log(q), 1)
        softmax_crossentropy_t(p, q) = crossentropy_t(p, softmax_t(q))
        shape = [5,10]
        a1 = genarg(shape, :ones, rand_offset=0)
        a2 = genarg(shape, :ones, rand_offset=0)

        @show test_output(softmax, softmax_t, (si1, a1))
        @show test_output(crossentropy, crossentropy_t, (ci1, a1), (ci2, a2))
        @show test_output(softmax_crossentropy, softmax_crossentropy_t, (sci1, a1), (sci2, a2))


        @show test_gradients(cg.sum(softmax), debug=true)
        @show test_gradients(cg.sum(crossentropy))
        @show test_gradients(cg.sum(softmax_crossentropy))
    end
end

function test_sgd_basics()
    show_test("sgd")
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
    show_test("sum")
    a = cg.placeholder([1])
    b = cg.variable(cg.randn(cg.constant([1,5])))
    c = cg.sum(a, cg.constant(1))
    d = c * b
    e = sum(d)
    test_gradients(e, [3, 5])
end


#test_broadcast()
#test_shape_gradients()
test_scalar_gradients()
test_other_gradients()
test_nn()
test_get_and_set_gradients()
test_sgd_basics()
test_sum()
