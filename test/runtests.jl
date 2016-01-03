using cg
using Base.Test

# Test each operator
function test_gradients(out::cg.Node, shape=[5])
    G = cg.get_graph([out]);
    inputs = filter(x -> isa(x.op, cg.Placeholder), G.nodes)
    inputs = collect(inputs)
    gradients = cg.grad(out, inputs)

    session = cg.Session(out)
    for input = inputs
        session.values[input] = 2 * ones(Float64, shape...) # Make random?
    end

    for (input,grad) = zip(inputs, gradients)
        numeric = cg.numeric_grad(session, out, input, 0.00001)
        symbolic = cg.interpret(session, grad)
        @show session.values[out]
        @show numeric
        @show symbolic
        @test_approx_eq_eps(symbolic, numeric, 0.0001)
    end
end

# WHY do I have to do cg.t but not cg.sum?

function test_gradients()
    i = () -> cg.placeholder([1]) # Shape doesn't matter yet
    @show test_gradients(sum(i()))
    @show test_gradients(sum(-i()))
    @show test_gradients(sum(i() + i()))
    @show test_gradients(sum(i() - i()))
    @show test_gradients(sum(i() .* i()))
    @show test_gradients(sum(exp(i())))
    @show test_gradients(sum(log(i())))
    @show test_gradients(sum(i() ./ i()))
    @show test_gradients(cg.t(i()) * i())
    @show test_gradients(sum(cg.sigmoid(i())))
    #@show test_gradients(sum(cg.relu(i())))
end

function test_sgd_basics()
    # Test basic optimization - minimize sum((b - a)^2)
    target = rand(10)
    a = cg.constant(target, "a")
    b = cg.variable(cg.constant(zeros(10)), "b") # Parameter to optimize
    c = b - a
    d = c .* c
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
d = c .* b
e = sum(d)
test_gradients(e, [3, 5])
end


test_gradients()
test_sgd_basics()
test_sum()
