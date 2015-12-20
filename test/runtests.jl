using cg
using Base.Test

# Test each operator
function testGradients(out::cg.Node)
    G = cg.get_graph([out]);
    inputs = filter(x -> isa(x.op, cg.Placeholder), G.nodes)
    gradients = cg.grad(out, inputs)

    arguments = Dict{cg.Node, cg.TensorValue}()
    for input = inputs
        arguments[input] = [1.0, 1.0, 1.0, 1.0] # Make random?
    end
    for (input,grad) = zip(inputs, gradients)
        #point = copy(arguments[input])
        numeric = cg.numeric_grad(out, input, arguments, 0.0001)
        symbolic = cg.interpret(grad, arguments)
        @show numeric
        @show symbolic
        @test_approx_eq_eps(symbolic, numeric, 0.0001)
    end
end

# WHY do I have to do cg.t but not cg.sum?

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


# Test basic optimization
a = i()
b = cg.variable(cg.constant([10])) # Parameter
c = b - a
d = c .* c
e = sum(d)
example = rand(10)
#O = cg.optimizeWrt(F, a, example, e, [b], 2)
#@test_approx_eq example O[b]
