# NEXT UP: Linear regression
# Few more basic op types, implement update in interpret, basic neural network!
# Reorganize, make more general, cleanup
# TODO: Figure out how to cleanly support scalars alongside 1x1 arrays
using Base
importall Base.Operators
importall Base

typealias Float Float32

###############
# Basic types #
###############

abstract Node
abstract VarType
abstract OpType

# TODO: Make this graph container more powerful / actually do something
## build better graph abstraction, avoid repeated code
immutable Graph
    nodes::Set{Node}
end

# The lack of mutually recursive types is annoying
# Use T <: Node and Apply{Variable} instead
# TODO: Is there a better way to do this?
type Apply{T <: Node} <: Node
    op::OpType
    inputs::Vector{T}
    output::T
    name::Nullable{AbstractString}
end

type Variable <: Node
    owner::Nullable{Apply}
    clients::Vector{Apply}
    data::VarType
    name::Nullable{AbstractString}
end

immutable Func
    graph::Graph
    outputs::Vector{Variable}
    inputs::Vector{Variable}
    defaults::Dict{Variable, AbstractArray}
end

# Accessors
# TODO: Are accessors considered good style instead of accessing directly?
# TODO: Remove redundancies and reorganize
inputs(n::Apply{Variable}) = n.inputs::Vector{Variable}
output(n::Apply{Variable}) = n.output::Variable

pred(n::Apply{Variable}) = n.inputs::Vector{Variable}
succ(n::Apply{Variable}) = [n.output]::Vector{Variable}

pred(n::Variable) = isnull(n.owner) ? [] : [get(n.owner)]
succ(n::Variable) = n.clients::Vector{Apply}

function Variable(data::VarType, name::AbstractString="")
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Variable(Nullable(), [], data, newName)
end

function Func(outputs::Vector{Variable})
    Func(get_graph(outputs), outputs, Vector{Variable}(), Dict{Variable, AbstractArray}())
end

function Func(outputs::Vector{Variable}, inputs::Vector{Variable})
    Func(get_graph(union(outputs, inputs)), outputs, inputs, Dict{Variable, AbstractArray}())
end

function name{T <: Node}(n::T, str::AbstractString)
    n.name = Nullable(str)
    n
end

function apply(op::OpType, inputs::Vector{Variable}, name::AbstractString="")
    #TODO: Is there a way to combine the var and apply creation? Perhaps an inner constructor?
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    var = Variable(TensorVal())
    apply = Apply{Variable}(op, inputs, var, newName)
    var.owner = apply
    for i in inputs
        push!(i.clients, apply)
    end
    var
end

####
# Variable Types
####

# TODO: Decide whether everything is a matrix or not
# SUPER TODO: Shape inference
#### SUPER TODO: Broadcasting (see ?broadcast)

abstract Tensor <: VarType

# Use for constants which we write out explictly.
# Define larger constant tensors (e.g. zeros, ones...) by ConstantOp(val, [shape])
# similar for random.
type TensorConstant <: Tensor
    value::AbstractArray
    shape::Vector{Int}
end

# Values provided as input
type Input <: Tensor
    #shape::Vector{Int}
end

# Values produced by Apply
type TensorVal <: Tensor
    #shape::Vector{Int}
end

## Variable initialization

abstract InitMethod

type ConstantInit <: InitMethod
    value::Real
end

function init(method::ConstantInit, shape::Vector{Int})
    zeros(Float, tuple(shape...))
end

# Values which are initialized and potentially updated
type TensorVar <: Tensor
    shape::Vector{Int}
    init::InitMethod
end

function init(var::TensorVar)
    init(var.init, var.shape)
end

function variable(shape::Vector{Int}, method::InitMethod, name::AbstractString="")
    Variable(TensorVar(shape, method), name)
end

function constant(val::Real, name::AbstractString="")
    Variable(TensorConstant([val], [1,1]), name)
end

function constant(val::AbstractArray, name::AbstractString="")
    Variable(TensorConstant(val, collect(size(val))), name)
end

# Specifies an input variable
function input(name::AbstractString="")
    Variable(Input(), name)
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

# Create
abstract CreateOp <: OpType
abstract ConstantOp <: CreateOp
abstract RandomOp <: CreateOp  # TODO: Make these!
abstract ElementWise <: OpType
abstract Activations <: ElementWise


type Fill <: ConstantOp end
type Zeros <: ConstantOp end
type Ones <: ConstantOp end
type OnesLike <: ConstantOp end

# .+ and + are different, just support + and - for now
type Add <: ElementWise end
type Sub <: ElementWise end
type Mul <: ElementWise end
type Div <: ElementWise end
type Neg <: ElementWise end

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

function gen_args(narg, typ::DataType)
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
function .*(a::Variable, b::Variable) apply(Mul(), [a, b]) end
"""
macro register_op(typ, op, narg)
    # TODO: Is there a way to interpolate an expr (like splat) into another expr with $ or similar?
    # For now, use Expr function (for which we can use splat).
    # Actually think it's pretty clear.
    args, apply_args = gen_args(narg, Variable)
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
function grad(op::Mul, ds::Variable, a::Variable, b::Variable)
    [a .* ds, b .* grad_out]
end
"""
macro register_grad(typ, grads...)
    args, _ = gen_args(length(grads), Variable)
    Expr(:function,
         Expr(:call,
              esc(:grad),
              :(op::$typ),
              :(ds::Variable),
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
    args, _ = gen_args(narg, AbstractArray)
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

@register_op Fill        fill         2
@register_op Zeros       zeros        1
@register_op Ones        ones         1
@register_op OnesLike    ones_like    1
@register_op Dim         dim          1

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

@register_impl Fill         2   Base.fill(a[1], round(Int64, b)...)
@register_impl Zeros        1   zeros(Float, Round(Int64, a)...)
@register_impl Ones         1   ones(Float, Round(Int64, a)...)
@register_impl OnesLike     1   Base.ones(a)
@register_impl Dim          1   collect(Int, size(a))

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
@register_grad Sum fill(ds, dim(a))


########################
# Gradient computation #
########################

# Numeric gradient of output with respect to `wrt`
function numeric_grad(f::Func, wrt::Variable, value::AbstractArray, eps=0.001)
    argValue = float(value)
    result = zeros(value)
    arg = Dict{Variable, AbstractArray}(wrt => argValue)
    for i in 1:length(argValue)
        argValue[i] += eps
        res1 = interpretRetArgs(f, arg)[1]
        argValue[i] -= 2eps
        res2 = interpretRetArgs(f, arg)[1]
        argValue[i] += eps
        @assert length(res1) == 1
        result[i] = (res1[1] - res2[1]) / 2eps
    end
    result
end

function grad(graph::Graph, out::Variable, wrt::Vector{Variable})
    # Set of nodes on all paths between the set `wrt` and `out`
    downstream = influenced_by(wrt, true)
    upstream = influenced_by([out], false)
    on_path = intersect(upstream, downstream)

    toposorted = toposort(graph)
    node_to_grad = Dict{Variable, Variable}(out => ones_like(out))
    for node = reverse(toposorted)
        if !(node in on_path)
            continue
        end
        if isa(node, Variable)
            # Variables should have grad calculated by the time we process them
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
    # Make sure they are all a part of it - TODO make this have less overhead
    for node = get_connected(graph.nodes)
        push!(graph.nodes, node)
    end
    node_to_grad
end


# Return set of nodes that are influenced in the DAG from any in set `nodes`
# i.e. that would be influenced by in a computation
# child=true means go to children in dag, false means go to parents
function influenced_by(nodes::Vector{Variable}, child::Bool)
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

# Super slow interpret functions
function interpret(f::Func, arguments::Tuple{AbstractArray})
    #@assert length(f.inputs) == length(arguments)
    args = Dict{Variable, AbstractArray}()
    for (input, arg) = zip(f.inputs, arguments)
        args[input] = arg
    end
    interpret(f, args)
end

# Return back list of output arguments in order
function interpretRetArgs(f::Func, state::Dict{Variable, AbstractArray})
    state = interpret(f, state)
    result = []
    for arg = f.outputs
        @assert haskey(state, arg)
        push!(result, get(state, arg, :impossible))
    end
    result
end

function initialize_function(f::Func)
    values = Dict{Variable,AbstractArray}()
    for node = f.graph.nodes
        if isa(node, Variable)
            if isa(node.data, TensorVar)
                values[node] = init(node.data)
            elseif isa(node.data, TensorConstant)
                values[node] = node.data.value
            end
        end
    end
    return values
end

# Takes dictionary mapping each already set variable to a state
# Will not overwrite constants/variables which are already present
# Return back dictionary representing current state
function interpret(f::Func, values::Dict{Variable, AbstractArray}=Dict{Variable,AbstractArray}())
    # Is something like T <: Real possible for AbstractArray in arguments?

    order = toposort(f.graph)
    for node = order
        if isa(node, Variable)
            if !(node in keys(values))
                if node in keys(f.defaults)
                    values[node] = f.defaults[node]
                elseif isa(node.data, Input)
                    @assert false && "Every input node must have a value"
                elseif isa(node.data, TensorVar)
                    # Initialize if not explicitly given
                    #values[node] = init(node.data)
                    @assert false && "Call initialize_function first"
                elseif isa(node.data, TensorConstant)
                    #values[node] = node.data.value
                    @assert false && "Call initialize_function first"
                elseif isa(node.data, TensorVal)
                    #print(tostring(node))
                    @assert false && "Every TensorVal node must have a parent"
                else
                    @assert false && "Unknown Variable type"
                end
            end
        elseif isa(node, Apply)
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
    return values
end

## TODO: Transform to straight Julia source code
# Time to learn some metaprogramming!
################
# Optimization #
################

function optimizeWrt(f, input::Variable, data::Array, loss::Variable, parameters::Vector{Variable}, max_steps::Int)
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

function get_graph(nodes::Vector{Variable})
    Graph(get_connected(Set(nodes)))
end

##################
# Representation #
##################

# Print connected component of node
function to_dot(node::Node)
    G = get_graph([node])
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
        shape = isa(node, Apply) ? "box" : "ellipse"
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

function tostring(node::Variable)
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.data))"
    else
        return "$(typeof(node.data))"
    end
end

function tostring(node::Apply)
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
