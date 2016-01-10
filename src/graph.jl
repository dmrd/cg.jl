# Defines the core Node abstraction
# Defines the core Node and operations on the graph

using Base
importall Base

typealias Float Float64
typealias TensorValue Union{Real, Array}

###############
# Basic types #
###############
abstract OpType

type Node
    op::OpType
    inputs::Vector{Node}
    outputs::Vector{Node}
    name::AbstractString
end

pred(node::Node) = node.inputs
succ(node::Node) = node.outputs

# TODO: Actual scoping on naming
function Node(data::OpType, name::AbstractString="")
    newName = length(name) == 0 ? "$(gensym())" : name
    Node(data, Vector{Node}(), Vector{Node}(), newName)
end

function Node(op::OpType, inputs::Vector{Node}, name::AbstractString="")
    newName = length(name) == 0 ? "$(gensym())" : name
    node = Node(op, inputs, Vector{Node}(), newName)
    for i in inputs
        push!(i.outputs, node)
    end
    node
end

immutable Graph
    nodes::Set{Node}
end

# May use to have a notion of "default" graph
# const Dict{Symbol, Graph} context
# context[:default] = Graph(Set{Node}())
########################
# Gradient computation #
########################

# out w.r.t. each element of wrt
function grad(out::Node, wrt::Vector{Node})
    # input gradients = derivative w.r.t. input - may be different for the nth and n+1th argument
    # output gradient = derivative w.r.t. output (i.e. before we apply this node's grad)

    # When we process a node, associate its input gradient with the appropriate input
    node_to_grad_vec = Dict{Node, Vector{Node}}(out => [ones_like(out)])

    # Map a node to its single output (usually sum of all output gradients)
    node_to_grad = Dict{Node, Node}()

    # Set of nodes on all paths between the set `wrt` and `out`
    for node = reverse(toposort_on_path([out], wrt))
        # Compute the gradient w.r.t. the node's output
        @assert haskey(node_to_grad_vec, node)
        grads = node_to_grad_vec[node]
        if (length(grads) == 1)
            output_grad = node_to_grad_vec[node][1]
        else
            # Sum up the gradients of all the outputs
            output_grad = foldr((sum, next) -> sum + next, grads)
        end
        if length(node.name) > 0
            output_grad.name = "Gin:$(node.name)"
        end
        node_to_grad[node] = output_grad


        if (length(node.inputs) > 0)
            # If the node has inputs, calculate the gradient w.r.t their outputs along this path
            input_grads = grad(node.op, node, output_grad, node.inputs...)
            for (input_node, grad) = zip(node.inputs, input_grads)
                if (!haskey(node_to_grad_vec, input_node))
                    node_to_grad_vec[input_node] = Vector{Node}()
                end
                if length(grad.name) > 0
                    output_grad.name = "Gout:$(input_node.name)"
                end
                push!(node_to_grad_vec[input_node], grad)
            end
        end
    end
    [node_to_grad[x] for x in wrt]
end


####################
# Graph operations #
####################


# Node hashing for use in deduping computation graphs
# Hash node itself, then recursively add in hashes of parents
function hash_node(node::Node)
    res = UInt64(node.op)
    for input = node.inputs
        res = hash(hash_node(input), res)
    end
    res
end

# Return set of nodes that are influenced in the DAG from any in set `nodes`
# i.e. that would be influenced by in a computation
# child=true means go to children in dag, false means go to parents
function influenced_by(nodes::Vector{Node}, child::Bool)
    queue = copy(nodes)
    influenced = Set{Node}(queue)
    next_method = child ? succ : pred
    while !isempty(queue)
        next = pop!(queue)
        for node = next_method(next)
            if node in influenced
                continue
            end
            push!(queue, node)
            push!(influenced, node)
        end
    end
    influenced
end


function nodes_on_path(sink::Vector{Node}, source::Vector{Node}=Vector{Node}())
    downstream = influenced_by(source, true)
    upstream = influenced_by(sink, false)
    if length(source) > 0
        intersect(upstream, downstream)
    else
        upstream
    end
     
end

function toposort_on_path(sink::Vector{Node}, source::Vector{Node}=Vector{Node}())
    nodes = nodes_on_path(sink, source)
    toposorted = toposort(get_graph(union(sink, source)))
    intersect(toposorted, nodes)
end


# Returns nodes in topological order
function toposort(graph::Graph)
    result = Vector{Node}()
    marks = Dict{Node, Symbol}() # :marked, :temp
    function visit(cur::Node)
        mark = get(marks, cur, :unmarked)
        if isequal(mark, :temp)
            @assert false && "Graph is not a DAG!"
            return
        elseif isequal(mark, :unmarked)
            marks[cur] = :temp
            for child = succ(cur)
                visit(child)
            end
            marks[cur] = :marked
            push!(result, cur)
        end
    end

    for node = graph.nodes
        if isequal(get(marks, node, :unmarked), :unmarked)
            visit(node)
        end
    end
    reverse!(result)
    result
end

# Return graph consisting of all nodes connected to given Variables
function get_connected(nodes::Set{Node})
    stack = collect(Node, nodes)
    seen = Set{Node}(nodes)
    while !isempty(stack)
        cur = pop!(stack)
        prev = pred(cur)
        next = succ(cur)
        # Would like the chain(.) function
        for n in prev
            if !(n in seen)
                push!(seen, n)
                push!(stack, n)
            end
        end
        for n in next
            if !(n in seen)
                push!(seen, n)
                push!(stack, n)
            end
        end
    end
    seen
end

function get_graph(nodes::Vector{Node})
    Graph(get_connected(Set(nodes)))
end


function group_nodes(nodes::Vector{Node}, group::AbstractString)
    for node in nodes
        node.name = string(group, "/", node.name)
    end
end

function group_between(inputs::Vector{Node}, outputs::Vector{Node}, group::AbstractString ; include_in=true, include_out=true)
    set = nodes_on_path(outputs, inputs)
    if !include_in
        set = setdiff(set, Set(inputs))
    end
    if !include_out
        set = setdiff(set, Set(outputs))
    end
    group_nodes(collect(set), group)
end
