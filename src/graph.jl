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

abstract Node
abstract VarType
abstract OpType


immutable Shape
    shape::Vector{Int}  # length >= 0
end

# The lack of mutually recursive types is annoying
# Use T <: Node and Operation{Tensor} instead
# TODO: Is there a better way to do this?
type Operation{T <: Node} <: Node
    op::OpType
    inputs::Vector{T}
    output::T
    name::Nullable{AbstractString}
end

type Tensor <: Node
    owner::Nullable{Operation}
    clients::Vector{Operation}
    data::VarType
    name::Nullable{AbstractString}
    flags::Vector{Tuple{Symbol, Symbol}}
    # Type?  All float for now
end

# TODO: Make this do a graph copy and precompute important values such as toposort
immutable Session
end

immutable Graph
    nodes::Set{Node}
end

# May use to have a notion of "default" graph
# const Dict{Symbol, Graph} context
# context[:default] = Graph(Set{Node}())

# Accessors
# TODO: Are accessors considered good style instead of accessing directly?
# TODO: Remove redundancies and reorganize
inputs(n::Operation{Tensor}) = n.inputs::Vector{Tensor}
output(n::Operation{Tensor}) = n.output::Tensor

pred(n::Operation{Tensor}) = n.inputs::Vector{Tensor}
succ(n::Operation{Tensor}) = [n.output]::Vector{Tensor}

pred(n::Tensor) = isnull(n.owner) ? [] : [get(n.owner)]
succ(n::Tensor) = n.clients::Vector{Operation}

# TODO: Actual scoping on naming
function Tensor(data::VarType, name::AbstractString="")
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Tensor(Nullable(), [], data, newName, [])
end

function name{T <: Node}(n::T, str::AbstractString)
    n.name = Nullable(str)
    n
end

function apply(op::OpType, inputs::Vector{Tensor}, name::AbstractString="")
    #TODO: Is there a way to combine the var and apply creation? Perhaps an inner constructor?
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    var = Tensor(Result())
    apply = Operation{Tensor}(op, inputs, var, newName)
    var.owner = apply
    for i in inputs
        push!(i.clients, apply)
    end
    var
end

####
# Possible Tensor node types
####

# TODO: Decide whether everything is a matrix or not
# SUPER TODO: Shape inference
#### SUPER TODO: Broadcasting (see ?broadcast)

# Use for constants which we write out explictly.
# Define larger constant tensors (e.g. zeros, ones...) by ConstantOp(val, [shape])
# similar for random.
# Unclear this is necessary
# type Constant <: Tensor
#     shape::Shape
#     value::Array
# end

# Values provided as input
immutable Placeholder <: VarType
    shape::Shape
end

# Some value that is initialized once (by owner) and shared across runs
immutable Variable <: VarType
end

# Values produced by Operation
immutable Result <: VarType
    #shape::Vector{Int}
end

function variable(init::Tensor, name::AbstractString="")
    # TODO: must have shape specified / be able to infer
    output = copy(init)
    output.data = Variable()
end

# Specifies an input variable
function placeholder(shape::Shape, name::AbstractString="")
    Tensor(Placeholder(shape), name)
end

####
# Operations
####

# TODO: Is there a better way than instantiating the op as the first argument?
# TODO: Does this type hierarchy make any sense?  Think carefully about what's necessary
# Also whether it needs a hierarchy at all.  Unclear if we make use of it anywhere
# Additionally, can these be defined together with other parts of command?
# Either reorganize or even have these created inside macro
# (i.e. pass in `Zeros <: ConstantOp` as a parameter - probably unnecessary)

# TODO: Make all ops that take arguments accept tensors as the argument

# Create
abstract CreateOp <: OpType
abstract ConstantOp <: CreateOp
abstract RandomOp <: CreateOp  # TODO: Make these!
abstract ElementWise <: OpType
abstract Activations <: ElementWise


immutable Zeros <: ConstantOp end
immutable ZerosLike <: ConstantOp end
immutable Ones <: ConstantOp end
immutable OnesLike <: ConstantOp end
immutable Fill <: ConstantOp end

immutable Constant <: ConstantOp
    value::TensorValue
end

function constant(value::TensorValue, name::AbstractString="")
    apply(Constant(value), Vector{Tensor}(), name)
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
function .*(a::Tensor, b::Tensor) apply(Mul(), [a, b]) end
"""
macro register_op(typ, op, narg)
    # TODO: Is there a way to interpolate an expr (like splat) into another expr with $ or similar?
    # For now, use Expr function (for which we can use splat).
    # Actually think it's pretty clear.
    args, apply_args = gen_args(narg, Tensor)
    Expr(:function,
         Expr(:call,
              esc(op),
              args...),
         Expr(:call,
              :apply,
              Expr(:call, typ),
              Expr(:vect, apply_args...)))
end

"""
@register_grad Mul (a .* ds) (b .* ds)
 Expands to
function grad(op::Mul, ds::Tensor, a::Tensor, b::Tensor)
    [a .* ds, b .* grad_out]
end
"""
macro register_grad(typ, grads...)
    args, _ = gen_args(length(grads), Tensor)
    Expr(:function,
         Expr(:call,
              esc(:grad),
              :(op::$typ),
              :(ds::Tensor),
              args...),
         Expr(:vect,
              grads...))
end

"""
@register_impl Mul 3 (a + b + c)
Expands to
function op(op::Mul, a::AbstractArray, b::AbstractArray, c::AbstractArray)
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

# Wrapper on fill
#fill(val, shape::Array{Int}, name::AbstractString="") = fill(constant(val), constant(shape))

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
@register_impl InPlaceAdd   2   (for i in 1:length(a); a[i] += b[i] end)
@register_impl Sum          1   [Base.sum(a)]  # I seriously need to handle reals

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
function numeric_grad(target::Tensor, wrt::Tensor, values::Dict{Tensor, TensorValue}, eps=0.001)
    arg = values[target]
    result = zeros(arg)
    for i in 1:length(arg)
        arg[i] += eps
        res1 = interpret(wrt, values)[1]
        arg[i] -= 2eps
        res2 = interpret(wrt, values)[1]
        arg[i] += eps
        @assert length(res1) == 1
        result[i] = (res1[1] - res2[1]) / 2eps
    end
    result
end

# out w.r.t. each element of wrt
function grad(out::Tensor, wrt::Vector{Tensor})
    # Set of nodes on all paths between the set `wrt` and `out`
    downstream = influenced_by(wrt, true)
    upstream = influenced_by([out], false)
    on_path = intersect(upstream, downstream)

    toposorted = toposort(graph)
    node_to_grad = Dict{Tensor, Tensor}(out => ones_like(out))
    for node = reverse(toposorted)
        if !(node in on_path)
            continue
        end
        if isa(node, Tensor)
            # Tensor should have grad calculated by the time we process them
            @assert haskey(node_to_grad, node)
        else
            # Should have already computed output's gradient
            @assert haskey(node_to_grad, node.output)
            gradients = grad(node.op, node_to_grad[node.output], inputs(node)...)
            for (original, gradient) in zip(inputs(node), gradients)
                if original in on_path
                    node_to_grad[original] = gradient
                    if !isnull(original.name)
                        name(gradient, "G:$(get(original.name))")
                    end
                    #original .= gradient
                end
            end
        end
    end
    [node_to_grad[x] for x in wrt]
end


# Return set of nodes that are influenced in the DAG from any in set `nodes`
# i.e. that would be influenced by in a computation
# child=true means go to children in dag, false means go to parents
function influenced_by(nodes::Vector{Tensor}, child::Bool)
    queue = Vector{Node}(nodes)
    influenced = Set{Node}(nodes)
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

function interpret(output::Node, values::Dict{Tensor, TensorValue}=Dict{Tensor,TensorValue}())
    interpret([output], values)[1]
end

# Takes dictionary mapping each already set variable to a state
# Will not overwrite constants/variables which are already present
# Return back dictionary representing current state
# TODO: Will want the ability to provide ops that feed a placeholder variable
function interpret{T <: Node}(outputs::Vector{T}, values::Dict{Tensor, TensorValue}=Dict{Tensor,TensorValue}())
    # TODO - function to go up from node 
    order = toposort(get_graph(outputs))
    for node = order
        if isa(node, Variable)
            if !(node in keys(values))
                if isa(node.data, Placeholder)
                    @assert false && "Every input node must have a value"
                elseif isa(node.data, Variable)
                    # Initialize if not explicitly given
                    @assert false && "All Variables should be preinitialized"
                elseif isa(node.data, Result)
                    #print(tostring(node))
                    @assert false && "Every Result node must have a parent"
                else
                    @assert false && "Unknown Variable type"
                end
            end
        elseif isa(node, Operation)
            args = []
            for arg = inputs(node)
                @assert haskey(values, arg)
                push!(args, get(values, arg, :impossible))
            end
            len = length(args)
            if len == 0
                out = op(node.op)
            elseif len == 1
                out = op(node.op, args[1])
            elseif len == 2
                out = op(node.op, args[1], args[2])
            else
                @assert "We have ops with more args now!?"
            end
            values[node.output] = out
        end
    end
    [values[output] for output in outputs]
end

## TODO: Transform to straight Julia source code
# Time to learn some metaprogramming!
################
# Optimization #
################
# TODO: Make more specific optimizers (SGD etc.)
function optimizeWrt(f, input::Variable, data::TensorValue, loss::Variable, parameters::Vector{Variable}, max_steps::Int)
    gradients = grad(f.graph, loss, parameters)

    state = initialize_function(f)
    state[input] = data
    for steps = 1:max_steps
        interpret(f, state)

        for param = parameters
            cur = state[param]
            update = state[gradients[param]]
            @show cur
            @show update
            @assert length(cur) == length(update)
            for i = 1:length(cur)
                cur[i] -= update[i]
            end
        end
    end
    #return map(x -> get(state, x, :ERROR), parameters)
    return state
end

####################
# Graph operations #
####################


# Node hashing for use in deduping computation graphs
function hashNode(node::Variable)
    # TODO: Cleaner chaining?
    if isnull(node.owner)
        return hash(node.data)
    else
        hash(node.owner, hash(data))
    end
end

function hashNode(node::Operation)
    hash(node.op, hash(inputs))
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
function get_connected{T <: Node}(nodes::Set{T})
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

function get_graph{T <: Node}(nodes::Vector{T})
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
        shape = isa(node, Operation) ? "box" : "ellipse"
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

function tostring(node::Variable)
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.data))"
    else
        return "$(typeof(node.data))"
    end
end

function tostring(node::Operation)
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.op))"
    else
        return "$(typeof(node.op))"
    end
end

function tostring{T <: Node}(nodes::Vector{T})
    c = ", "
    "[$(join(map(tostring, nodes), c))]"
end

# Pretty print a computation
function pprint(g::Graph)

end
