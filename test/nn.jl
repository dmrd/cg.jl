# Simple logistic
using cg

function softmax_regression(input_dim::Int, output_dim::Int, step_size::Float64)
    # Wow that's verbose - start overloading so constants become nodes automatically
    W1 = cg.variable(cg.randn(cg.constant([output_dim, input_dim])) ./ cg.constant(sqrt(input_dim)), "W1")
    b1 = cg.variable(cg.zeros(cg.constant(output_dim)), "b1")

    input = cg.placeholder([input_dim], "input");
    label = cg.placeholder([output_dim], "label");
    output = cg.softmax((W1 * input) + b1)
    loss = cg.crossentropy(label, output)

    train = cg.sgd_optimizer(loss, [W1, b1], cg.constant(step_size))

    (input, label, output, loss, train)
end

function simple_nn(input_dim::Int, hidden_dim::Int, output_dim::Int, step_size::Float64)
    W1 = cg.variable(cg.randn(cg.constant([hidden_dim,input_dim])) ./ cg.constant(sqrt(input_dim)), "W1")
    b1 = cg.variable(cg.zeros(cg.constant(hidden_dim)), "b1")
    W2 = cg.variable(cg.randn(cg.constant([output_dim, hidden_dim])) ./ cg.constant(sqrt(hidden_dim)), "W1")
    b2 = cg.variable(cg.zeros(cg.constant(output_dim)), "b2")

    input = cg.placeholder([input_dim], "input");
    label = cg.placeholder([1], "label");
    hidden_act = cg.sigmoid((W1 * input) + b1, "activation")
    output = cg.softmax((W2 * hidden_act) + b2)
    loss = crossentropy(label, output)

    train = sgd_optimizer(loss, [W1, W2, b1, b2], cg.constant(step_size))

    (input, label, output, loss, train)
end
