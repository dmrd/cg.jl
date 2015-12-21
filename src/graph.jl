# IN PROGRESS: Have only operations as a graph type

# Few more basic op types, basic neural network!
# Reorganize, make more general, cleanup
# TODO: Figure out how to cleanly support scalars alongside 1x1 arrays
using Base
importall Base.Operators
importall Base

typealias Float Float32
typealias TensorValue Union{Real, Array}

###############
# Basic types #
###############

typealias Shape Vector{Int}
abstract OpType

# IN PROGRESS: REPLACE NODES T

# The lack of mutually recursive types is annoying
# Use T <: Node and Operation{Tensor} instead
# TODO: Is there a better way to do this?
type Node
    op::OpType
    inputs::Vector{Node}
    outputs::Vector{Node}
    name::Nullable{AbstractString}
    #flags::Vector{Tuple{Symbol, Symbol}}
end

pred(node::Node) = node.inputs
succ(node::Node) = node.outputs

# TODO: Make this do a graph copy and precompute important values such as toposort
immutable Session
    initialized::Bool
    values::Dict{Node, TensorValue}
end

immutable Graph
    nodes::Set{Node}
end

# May use to have a notion of "default" graph
# const Dict{Symbol, Graph} context
# context[:default] = Graph(Set{Node}())

# TODO: Actual scoping on naming
function Node(data::OpType, name::AbstractString="")
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Node(data, Vector{Node}(), Vector{Node}(), newName)
end

function Node(op::OpType, inputs::Vector{Node}, name::AbstractString="")
    #TODO: Is there a way to combine the var and apply creation? Perhaps an inner constructor?
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    node = Node(op, inputs, Vector{Node}(), newName)
    for i in inputs
        push!(i.outputs, node)
    end
    node
end

function name(n::Node, str::AbstractString)
    n.name = Nullable(str)
    n
end


####
# Operations
####

# TODO: Decide whether everything is a matrix or not
# TODO: Shape inference
#### TODO: Broadcasting (see ?broadcast)
# TODO: Is there a better way than instantiating the op as the first argument?
# TODO: Does this type hierarchy make any sense?  Think carefully about what's necessary
# Also whether it needs a hierarchy at all.  Unclear if we make use of it anywhere
# Additionally, can these be defined together with other parts of command?

abstract VarOp <: OpType
abstract CreateOp <: OpType
abstract ConstantOp <: CreateOp
abstract RandomOp <: CreateOp  # TODO: Make these!
abstract ElementWise <: OpType
abstract Activations <: ElementWise

# TODO: How to do control edges / grouping
immutable Noop <: OpType end

immutable Zeros <: ConstantOp end
immutable ZerosLike <: ConstantOp end
immutable Ones <: ConstantOp end
immutable OnesLike <: ConstantOp end
immutable Fill <: ConstantOp end

# Values provided as input
immutable Placeholder <: VarOp
    shape::Shape
end

# Some value that is initialized once (on first run) and shared across runs
immutable Variable <: VarOp
    init::Node
end

immutable Constant <: ConstantOp
    value::TensorValue
end

# Specifies a mutable value
function variable(init::Node, name::AbstractString="")
    # TODO: must have shape specified / be able to infer
    # TODO: Actually initialize variables once in a session
    Node(Variable(init), name)
end

# Specifies an input variable
function placeholder(shape::Shape, name::AbstractString="")
    Node(Placeholder(shape), name)
end

function constant(value::TensorValue, name::AbstractString="")
    Node(Constant(value), name)
end

# .+ and + are different, just support + and - for now
type Add <: ElementWise end
type Sub <: ElementWise end
type Mul <: ElementWise end
type Div <: ElementWise end
type Neg <: ElementWise end
type Copy <: ElementWise end

type Sigmoid <: Activations end
type Relu <: Activations end

type SoftMax <: OpType end

type MatMul <: OpType end  # Matrix multiply
type Transpose <: OpType end
type Sum <: OpType end
type Dim <: OpType end
type Assign <: OpType end
type InPlaceAdd <: OpType end



#############################
# Operation creation macros #
#############################

# Each operation needs:
## 1. Type (identifier)
## 2. functions (call or infix, e.g. add and +)
## 4. Implementation (CPU/GPU)
## 3. gradients
## 4. shape inference


# TODO: make it easy to define parts of an operation all together?
# e.g. @createOp Add, add, [a, b], [g, g], a + b, [shape inference]
#                type, name, args, gradients, implementation, [shape inference]

function gen_args(narg, typ::Type)
    assert(narg >= 0)
    args = []
    apply_args = []
    # Var names 'a'...'z'
    for var = 'a':('a' + narg - 1)
        varsym = symbol(var)
        push!(args, :($(varsym)::$(typ)))
        push!(apply_args, varsym)
    end
    args, apply_args
end

"""
@register_op  Mul (.*) 2
Expands to
function .*(a::Node, b::Node) apply(Mul(), [a, b]) end
"""
macro register_op(typ, op, narg)
    # TODO: Is there a way to interpolate an expr (like splat) into another expr with $ or similar?
    # For now, use Expr function (for which we can use splat).
    # Actually think it's pretty clear.
    args, apply_args = gen_args(narg, Node)
    Expr(:function,
         Expr(:call,
              esc(op),
              args...,
              ),
         #
         Expr(:call,
              :Node,
              Expr(:call, typ),
              Expr(:vect, apply_args...)))
    # Expr to add in optional name argument
    # 4: Expr
    #   head: Symbol kw
    #   args: Array(Any,(2,))
    #     1: Expr
    #       head: Symbol ::
    #       args: Array(Any,(2,))
    #       typ: Any
    #     2: ASCIIString ""
end

"""
@register_grad Mul (a .* ds) (b .* ds)
 Expands to
function grad(op::Mul, ds::Node, a::Node, b::Node)
    [a .* ds, b .* grad_out]
end
"""
macro register_grad(typ, grads...)
    args, _ = gen_args(length(grads), Node)
    Expr(:function,
         Expr(:call,
              esc(:grad),
              :(op::$typ),
              :(ds::Node),
              args...),
         Expr(:vect,
              grads...))
end

"""
@register_impl Mul 3 (a + b + c)
Expands to
function op(op::Mul, a::TensorValue, b::TensorValue, c::TensorValue)
    a + b + c
end
"""
macro register_impl(typ, narg, impl)
    args, _ = gen_args(narg, TensorValue)
    Expr(:function,
         Expr(:call,
              esc(:op),
              :(op::$typ),
              args...),
         impl)
end

#########################
# Operation Definitions #
#########################
# TODO: Define all parts of an operation together or keep similar parts grouped?
# Todo ops:
    # boolean operators
    # get/setindex
    # random
    # max/min
    # common pointwise math (e.g. exp)

# noop basics
function noop(a::Node...)
    Node(Noop(), collect(a))
end

# Just have it return a scalar until theres some notion of ordering
function op(op::Noop, a::TensorValue...)
    1
end

# Wrapper on fill
fill(val::Float, shape::Array{Int}, name::AbstractString="") = fill(constant(val), constant(shape))

@register_op Zeros       zeros        1
@register_op ZerosLike   zeros_like   1
@register_op Ones        ones         1
@register_op OnesLike    ones_like    1
@register_op Fill        fill         2
@register_op Shape       shape        1
@register_op Copy        copy         1

@register_op Add         (+)          2
@register_op Sub         (-)          2
@register_op Mul         (.*)         2
@register_op Div         (./)         2
@register_op Neg         (-)          1
@register_op MatMul      (*)          2
@register_op Transpose   t            1
@register_op Assign      (.=)         2
@register_op Sum         sum          1
@register_op InPlaceAdd  plusequals   2  # += doesn't work

@register_op Sigmoid     sigmoid      1
#@register_op Relu        relu         1

#@register_op SoftMax     softmax      1

####

# TODO: use `call` instead

@register_impl Constant     0   op.value
# a = scalar, b = 1d array
@register_impl Fill         2   Base.fill(a, b...)
@register_impl Zeros        1   zeros(Float, a...)
@register_impl ZerosLike    1   Base.zeros(a)
@register_impl Ones         1   ones(Float, a...)
@register_impl OnesLike     1   Base.ones(a)
@register_impl Dim          1   collect(Int, size(a))

@register_impl Copy         1   a

@register_impl Add          2   a .+ b
@register_impl Sub          2   a .- b
@register_impl Mul          2   a .* b
@register_impl Div          2   a ./ b
@register_impl Neg          1   (-a)
@register_impl MatMul       2   a * b
@register_impl Transpose    1   transpose(a)
# Return scalar for now
@register_impl InPlaceAdd   2   (for i in 1:length(a); a[i] += b[i] end; 1)

# I seriously need to handle reals - should this be len 1 matrix or a scalar?
@register_impl Sum          1   [Base.sum(a)]

# Could do in terms of basic ops
@register_impl Sigmoid      1    (1.0 ./ (1.0 + exp(-a))) 
#@register_impl Relu         1    max(0, a)

####

@register_grad Add ds ds
@register_grad Sub (ds) (-ds)
@register_grad Mul (b .* ds) (a .* ds)
@register_grad Div (ds ./ b) (ds .* a)
@register_grad Neg -ds
@register_grad MatMul (ds * t(b)) (t(a) * ds)
@register_grad Transpose ds
@register_grad Sigmoid (sigmoid(a) .* (ones_like(a) - sigmoid(a)) .* ds)
#@register_grad Relu ((a .> zero(a[1])) .* ds)
@register_grad Sum ds .* ones_like(a)  # Only true if output is scalar

# TODO How to treate OnesLike etc. in gradient computations?


########################
# Gradient computation #
########################

# Numeric gradient of output with respect to `wrt`


function numeric_grad(target::Node, wrt::Node, values::Dict{Node, TensorValue}, eps=0.001)
    arg = values[wrt]
    result = zeros(arg)
    for i in 1:length(arg)
        arg[i] += eps
        res1 = copy(interpret(target, values))
        arg[i] -= 2eps
        res2 = copy(interpret(target, values))
        arg[i] += eps
        @assert length(res1) == 1
        @assert length(res2) == 1
        result[i] = (res1[1] - res2[1]) / 2eps
    end
    result
end

# out w.r.t. each element of wrt
function grad(out::Node, wrt::Vector{Node})
    # Set of nodes on all paths between the set `wrt` and `out`
    downstream = influenced_by(wrt, true)
    upstream = influenced_by([out], false)
    on_path = intersect(upstream, downstream)

    toposorted = toposort(get_graph([out]))

    # input gradients = derivative w.r.t. input - may be different for the nth and n+1th argument
    # output gradient = derivative w.r.t. output (i.e. before we apply this node's grad)

    # When we process a node, associate its input gradient with the appropriate input
    node_to_grad_vec = Dict{Node, Vector{Node}}()

    # Map a node to its single output (usually sum of all output gradients)
    node_to_grad = Dict{Node, Node}()

    node_to_grad_vec[out] = [ones_like(out)];
    for node = reverse(toposorted)
        if !(node in on_path)
            continue
        end

        # Compute the gradient w.r.t. the node's output
        @assert haskey(node_to_grad_vec, node)
        grads = node_to_grad_vec[node]
        if (length(grads) == 1)
            output_grad = node_to_grad_vec[node][1]
        else
            # Sum up the gradients of all the outputs
            output_grad = foldr((sum, next) -> sum + next, grads)
        end
        if !isnull(node.name)
            name(output_grad, "Gin:$(get(node.name))")
        end
        node_to_grad[node] = output_grad


        if (length(node.inputs) > 0)
            # If the node has inputs, calculate the gradient w.r.t their outputs along this path
            input_grads = grad(node.op, output_grad, node.inputs...)
            for (input_node, grad) = zip(node.inputs, input_grads)
                if (!haskey(node_to_grad_vec, input_node))
                    node_to_grad_vec[input_node] = Vector{Node}()
                end
                if !isnull(grad.name)
                    name(output_grad, "Gout:$(get(input_node.name))")
                end
                push!(node_to_grad_vec[input_node], grad)
            end
        end
    end
    [node_to_grad[x] for x in wrt]
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

###################
# Interpret Graph #
###################

function interpret(output::Node, values::Dict{Node, TensorValue}=Dict{Node, TensorValue}())
    interpret([output], values)[1]
end

# Takes dictionary mapping each already set variable to a state
# Will not overwrite constants/variables which are already present
# Return back dictionary representing current state
# TODO: Will want the ability to provide ops that feed a placeholder variable
function interpret(outputs::Vector{Node}, values::Dict{Node, TensorValue}=Dict{Node,TensorValue}())
    # TODO - function to go up from node instead of evaluating entire thing! use on_path?
    order = toposort(get_graph(outputs))
    for node = order
        # Handle various input types separately from normal
        if isa(node.op, Placeholder)
            # TODO: Not the actual behavior of placeholders - should have a loader of some kind
            if !haskey(values, node)
                @assert false && "Every input node must have a value"
            end
        elseif isa(node.op, Variable)
            if !haskey(values, node)
                # lol recursion - could be a bad time later
                values[node] = interpret(node.op.init, values)
            end
        elseif isa(node.op, Constant)
            if !(node in keys(values))
                values[node] = node.op.value
            end
        else
            args = Vector{TensorValue}()
            for arg = node.inputs
                @assert haskey(values, arg)
                push!(args, get(values, arg, :impossible))
            end
            len = length(args)
            values[node] = op(node.op, args...)
        end
    end
    [values[output] for output in outputs]
end

## TODO: Transform to straight Julia source code
# Time to learn some metaprogramming!
################
# Optimization #
################

# Create an optimize op and return 
function sgdOptimizer(loss::Node, variables::Vector{Node}, step_size::Node)
    gradients = grad(loss, variables)
    step_sizes = map(grad -> step_size .* grad, gradients)
    updates = map(vargrad -> plusequals(vargrad[1], (-step_size .* vargrad[2])), zip(variables, gradients))
    noop(updates...)
end

####################
# Graph operations #
####################


# Node hashing for use in deduping computation graphs
# Hash node itself, then recursively add in hashes of parents
function hashNode(node::Node)
    res = UInt64(node.op)
    for input = node.inputs
        res = hash(hash(input), res)
    end
    res
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

##################
# Representation #
##################

# Print connected component of node
function to_dot(G::Graph)
    nodeIds = Dict{Node, Int}()
    id = 0
    for node in G.nodes
        nodeIds[node] = id
        id += 1
    end

    labels = Vector{AbstractString}()
    edges = Vector{AbstractString}()
    for node in G.nodes
        thisId = nodeIds[node]
        #shape = isa(node, Operation) ? "box" : "ellipse"
        shape = "ellipse"
        labelLine = string(thisId, " [shape=\"", shape,"\", label=\"", tostring(node), "\"];")
        push!(labels, labelLine)
        for next in succ(node)
            edge = "$(nodeIds[node]) -> $(nodeIds[next]);"
            push!(edges, edge)
        end
    end

    string("digraph computation {\n",
           join(labels,"\n"),
           "\n",
           join(edges,"\n"),
           "\n}"
           )
end

function render(G::Graph, outfile::AbstractString)
    dotstr = to_dot(G)
    dotcmd = `dot -Tpng -o $(outfile)`
    run(pipeline(`echo $(dotstr)`, dotcmd))
end

function tostring(node::Node)
    # TODO - include values for constants
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.op))"
    else
        return "$(typeof(node.data))"
    end
end

function tostring(nodes::Vector{Node})
    c = ", "
    "[$(join(map(tostring, nodes), c))]"
end

# Pretty print a computation
function pprint(g::Graph)

end
