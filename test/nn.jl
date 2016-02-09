# Simple example softmax function with associated utils
# Doesn't do a good job of making util function reusable
using cg
using MNIST

using ImageMagick
using Images

function softmax_regression(input_dim::Int, output_dim::Int, step_size::Float64)
    # Wow that's verbose - start overloading so constants become nodes automatically
    W1 = cg.variable(cg.zeros(cg.constant([output_dim, input_dim])), "W1")
    b1 = cg.variable(cg.zeros(cg.constant(output_dim)), "b1")

    input = cg.placeholder([input_dim], "input")
    label = cg.placeholder([output_dim], "label")
    unnormed = cg.broadcast("+", cg.matmul(W1, input), b1)
    prediction = cg.softmax_stable(unnormed)
    loss = cg.crossentropy(label, prediction)

    train = cg.sgd_optimizer(loss, [W1, b1], cg.constant(step_size))

    (input, label, prediction, loss, train)
end

function onehot(labels::AbstractArray; min::Int = 0, max::Int = 9)
    labels = round(Int, labels)
    result = zeros(Float64, max - min + 1, length(labels))
    for (i, label) in enumerate(labels)
        result[label - min + 1, i] = 1.0
    end
    result
end

function shuffle_data(data, labels)
    perm = shuffle(collect(1:size(labels, 2)))
    data[:, perm], labels[:, perm]
end

# labels = 1hot encoded
function train_steps(sess, batch_size, steps, data, labels)
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

function compute_loss(session, loss, data, labels)
    session.values[input] = data
    session.values[label] = labels
    cg.interpret(session, loss)
end

function visualize_filters(filters, w, h)
    output = zeros(h, w * size(filters,1))
    for i in 1:size(filters,1)
        start = (i-1) * h
        filter = filters[i, :]
        high = maximum(filter)
        low = minimum(filter)
        scaled = (filter - low) / (high - low)
        output[:, start+1:start+w] = scaled
    end
    convert(Image, output)
end

function accuracy(data, labels)
    sess.values[input] = data
    result = cg.interpret(sess, output)
    correct = findmax(labels, 1)[2]
    prediction = findmax(result, 1)[2]
    sum(correct .== prediction) / length(correct)
end

# Scale the data to [0,1] to avoid overflow
test_raw = testdata()
train_raw = traindata()
test = (test_raw[1] / 255.0, test_raw[2])
train = (train_raw[1] / 255.0, train_raw[2])

input, label, output, loss, sgd = softmax_regression(784, 10, 0.01)
sess = cg.Session(sgd)

if !isinteractive()
    @show compute_loss(sess, loss, train[1], onehot(train[2]))
    @show compute_loss(sess, loss, test[1], onehot(test[2]))
    train_steps(100, 1000, train[1], onehot(train[2]))
    @show compute_loss(sess, loss, train[1], onehot(train[2]))
    @show compute_loss(sess, loss, test[1], onehot(test[2]))
end
