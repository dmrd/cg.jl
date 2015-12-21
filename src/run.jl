immutable Session
    nodes::Vector{Node}  # Parts of graph stored in topological order
    # [x,y,z] => precomputed order to compute nodes to get x,y, and z
    evalOrder::Dict{Vector{Node}, Vector{Node}}
    values::Dict{Node, TensorValue}

    # TODO: How to handle (a) groups, and (b) initialization variables
    function Session(output::Node)
        # Nodes at time this session was created
        nodes = toposort(get_graph([output]))
        new(nodes, Dict{Node, Vector{Node}}(), Dict{Node, TensorValue}())
    end
end

# Get order to evaluate the graph in.  Takes nodes from the graph which initialized the session
function getOrder(session::Session, outputs::Vector{Node})
    if !haskey(session.evalOrder, outputs)
        session.evalOrder[outputs] = toposort_on_path(outputs)
    end
    session.evalOrder[outputs]
end

## TODO: Transform to straight Julia source code
# Time to learn some metaprogramming!

function numeric_grad(session::Session, target::Node, wrt::Node, eps=0.001)
    arg = session.values[wrt]
    isScalar = typeof(arg) <: Real
    if isScalar
        session.values[wrt] = arg + eps
        res1 = copy(interpret(session, target))
        session.values[wrt] = arg - eps
        res2 = copy(interpret(session, target))
        session.values[wrt] = arg
        @assert length(res1) == 1
        @assert length(res2) == 1
        result = (res1[1] - res2[1]) / 2eps
    else
        result = zero(arg)
        for i in 1:length(arg)
            arg[i] += eps
            res1 = interpret(session, target)
            arg[i] -= 2eps
            res2 = interpret(session, target)
            arg[i] += 2eps
            @assert length(res1) == 1
            @assert length(res2) == 1
            result[i] = (res1[1] - res2[1]) / 2eps
        end
    end
    result
end

#############
# NN layers #
#############

# C = input_dim, R = output_dim
# input = Cx1
# weights = RxC
# function fully_connected(input::Node, input_dim::Int, output_dim::Int, weightInit, biasInit)
#     (variable(init) * input) + variable(biasInit)
# end

###################
# Interpret Graph #
###################

function interpret(session::Session, output::Node)
    interpret(session, [output])[1]
end

# Takes dictionary mapping each already set variable to a state
# Will not overwrite constants/variables which are already present
# Return back dictionary representing current state
# TODO: Will want the ability to provide ops that feed a placeholder variable
function interpret(session::Session, outputs::Vector{Node})
    for node in getOrder(session, outputs)
        # Handle various input types separately from normal
        if isa(node.op, Placeholder)
            # TODO: Not the actual behavior of placeholders - should have a loader of some kind
            if !haskey(session.values, node)
                @assert false && "Every input node must have a value"
            end
        elseif isa(node.op, Variable)
            if !haskey(session.values, node)
                # lol recursion - could be a bad time later
                session.values[node] = interpret(session, node.op.init)
            end
        elseif isa(node.op, Constant)
            if !(node in keys(session.values))
                session.values[node] = node.op.value
            end
        else
            args = Vector{TensorValue}()
            for arg = node.inputs
                @assert haskey(session.values, arg)
                push!(args, get(session.values, arg, :impossible))
            end
            len = length(args)
            session.values[node] = node.op(args...)
        end
    end
    [session.values[output] for output in outputs]
end
