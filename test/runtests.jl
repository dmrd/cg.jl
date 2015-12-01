using cg
using Base.Test

# TODO: Generate gradient checks for all operators

# Very basic sanity check
a = cg.input("a");
b = cg.constant([3,2,1], "b")
c = cg.constant([10,15,20], "c")
d = a + b
e = cg.t(d) * c
f = sum(e)
g = cg.get_graph([c])
Q = cg.grad(g, f, [a, b, c])
f = cg.Func([e, Q[a]], [a])
res = cg.interpret(f, ([6,6,6],))[2]
numeric = cg.numeric_grad(f, [6.0,6.0,6.0], 0.0001)

@test_approx_eq_eps(res, numeric, 0.001)
