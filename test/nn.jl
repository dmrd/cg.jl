# Simple logistic
using cg
using MNIST

function softmax_regression(input_dim::Int, output_dim::Int, step_size::Float64)
    # Wow that's verbose - start overloading so constants become nodes automatically
    W1 = cg.variable(cg.zeros(cg.constant([output_dim, input_dim])), "W1")
    b1 = cg.variable(cg.zeros(cg.constant(output_dim)), "b1")

    input = cg.placeholder([input_dim], "input");
    label = cg.placeholder([output_dim], "label");
    unnormed = (W1 * input) + b1
    output = cg.softmax(unnormed)
    loss = cg.crossentropy(label, output)

    train = cg.sgd_optimizer(loss, [W1, b1], cg.constant(step_size))

    (input, label, output, loss, train)
end

function onehot(labels)
    labels = round(Int, labels)
    min = minimum(labels)
    max = maximum(labels)
    result = zeros(Int, max - min + 1, length(labels))
    for (i, label) in enumerate(labels)
        result[label - min + 1, i] = 1
    end
    result
end
test = testdata()
train = traindata()
input, label, output, loss, sgd = softmax_regression(784, 10, 0.01)

sess = cg.Session(sgd)

function shuffle_data(data, labels)
    perm = shuffle(collect(1:size(labels, 2)))
    data[:, perm], labels[:, perm]
end

# labels = 1hot encoded
function train_steps(batch_size, steps, data, labels)
    @assert size(data, 2) == size(labels, 2)
    @assert batch_size <= size(labels, 2)
    sdata, slabel = shuffle_data(data, labels)
    tsteps = 0
    bs = 1
    while tsteps < steps
        be = bs + batch_size - 1
        # Reshuffle and startover
        if be > size(labels, 2)
            sdata, slabel = shuffle(data, labels)
            bs = 1
            continue
        end
        tsteps += 1
        sess.values[input] = sdata[:, bs:be]
        sess.values[label] = slabel[:, bs:be]
        cg.interpret(sess, sgd)
    end
end

function test_loss()
    sess.values[input] = test[1]
    sess.values[label] = onehot(test[2])
    cg.interpret(sess, loss)
end

function train_loss()
    sess.values[input] = train[1]
    sess.values[label] = onehot(train[2])
    cg.interpret(sess, loss)
end

labels = onehot(testdata()[2])
