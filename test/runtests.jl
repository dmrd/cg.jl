using cg
using Base.Test

# Test each operator
function testGradients(out::cg.Node)
    G = cg.get_graph([out]);
    inputs = filter(x -> isa(x.op, cg.Placeholder), G.nodes)
    inputs = collect(inputs)
    gradients = cg.grad(out, inputs)

    arguments = Dict{cg.Node, cg.TensorValue}()
    for input = inputs
        arguments[input] = [1.0, 1.0, 1.0, 1.0] # Make random?
    end

    for (input,grad) = zip(inputs, gradients)
        numeric = cg.numeric_grad(out, input, arguments, 0.0001)
        symbolic = cg.interpret(grad, arguments)
        @show numeric
        @show symbolic
        @test_approx_eq_eps(symbolic, numeric, 0.0001)
    end
end

# WHY do I have to do cg.t but not cg.sum?

function testGradients()
    i = () -> cg.placeholder([1]) # Shape doesn't matter yet
    @show testGradients(sum(i()))
    @show testGradients(sum(-i()))
    @show testGradients(sum(i() + i()))
    @show testGradients(sum(i() - i()))
    @show testGradients(sum(i() .* i()))
    #@show testGradients(sum(i() ./ i()))
    @show testGradients(cg.t(i()) * i())
    @show testGradients(sum(cg.sigmoid(i())))
    #@show testGradients(sum(cg.relu(i())))
end

function testSgdBasic()
    # Test basic optimization - minimize sum((b - a)^2)
    target = rand(10)
    a = cg.constant(target, "a")
    b = cg.variable(cg.constant(zeros(10)), "b") # Parameter to optimize
    c = b - a
    d = c .* c
    e = sum(d)
    values = Dict{cg.Node, cg.TensorValue}()
    optimizer = cg.sgdOptimizer(e, [b], cg.constant(0.001, "step_size"))
    cg.render(cg.get_graph([b]), "graph.png")
    for i = 1:10000
        cg.interpret(optimizer, values)
        if values[e][1] <= 0.001
            break
        end
    end
    @test_approx_eq_eps target values[b] 0.1
end

testGradients()
testSgdBasic()
