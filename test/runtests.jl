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
        symbolic = cg.interpret(F)
        @show numeric
        @show symbolic
        @test_approx_eq_eps(symbolic[2], numeric, 0.0001)
    end
end

# WHY do I have to do cg.t but not cg.sum?

i = cg.input
@which sum(i())
@show testGradients(sum(i()))
@show testGradients(sum(-i()))
@show testGradients(sum(i() + i()))
#@show testGradients(sum(i() - i()))
@show testGradients(sum(i() .* i()))
#@show testGradients(sum(i() ./ i()))
@show testGradients(cg.t(i()) * i())
#testGradients(sum(cg.sigmoid(i())))
#testGradients(sum(cg.relu(i())))

# Very basic sanity check
#a = cg.input("a");
#b = cg.constant([1,1,1], "b")
#c = cg.constant([2, 2, 2], "c")
#d = a + b
#e = d .* c
#f = e - a
#g = e ./ f
#h = -g
#result = sum(h)
#graph = cg.get_graph([result])
##grads = cg.grad(graph, result, [a, b, c, d, e, f, g, h])
#grads = cg.grad(graph, result, [a])
#F = cg.Func([result, grads[a]], [a])
#
#symbolic = cg.interpret(F, ([6,6,6],))[2]
#numeric = cg.numeric_grad(F, [6.0,6.0,6.0], 0.0001)
#@show symbolic
#@show numeric
#

#a = cg.input("a");
#b = cg.constant([3,2,1], "b")
#c = cg.constant([10,15,20], "c")
#d = a + b
#e = d .* c
#e2 = a .* e
#f = sum(e)
#g = cg.get_graph([c])
#Q = cg.grad(g, f, [a, b, c])
#f = cg.Func([e, Q[a]], [a])
#res = cg.interpret(f, ([6,6,6],))[2]
#numeric = cg.numeric_grad(f, [6.0,6.0,6.0], 0.0001)
#@show res
#@show numeric
#
#@test_approx_eq_eps(res, numeric, 0.001)
