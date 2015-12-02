using cg
using Base.Test

# Test each operator
function testGradients(out::cg.Variable)
    G = cg.get_graph([out]);
    inputs = filter(x -> isa(x, cg.Variable) && isa(x.data, cg.Input), G.nodes)
    inputs = convert(Array{cg.Variable, 1}, collect(inputs))
    gradients = cg.grad(G, out, collect(cg.Variable, inputs))

    defaults = Dict{cg.Variable, AbstractArray}()
    for input = inputs
        defaults[input] = [1.0, 1.0, 1.0, 1.0] # Make random?
    end
    for input = inputs
        F = cg.Func(G, [out, gradients[input]], inputs, defaults)
        point = copy(defaults[input])
        numeric = cg.numeric_grad(F, input, point, 0.0001)
        symbolic = cg.interpret(F)[gradients[input]]
        @show numeric
        @show symbolic
        @test_approx_eq_eps(symbolic, numeric, 0.0001)
    end
end

# WHY do I have to do cg.t but not cg.sum?

i = cg.input
@which sum(i())
@show testGradients(sum(i()))
@show testGradients(sum(-i()))
@show testGradients(sum(i() + i()))
@show testGradients(sum(i() - i()))
@show testGradients(sum(i() .* i()))
#@show testGradients(sum(i() ./ i()))
@show testGradients(cg.t(i()) * i())
@show testGradients(sum(cg.sigmoid(i())))
#@show testGradients(sum(cg.relu(i())))
