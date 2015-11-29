using Base
importall Base.Operators

typealias Float Float32

######
# Basic types
#####

abstract Node
abstract VarType
abstract OpType

# TODO: Make this graph container more powerful / actually do something
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

function Variable(data::VarType, name::AbstractString="")
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Variable(Nullable(), [], data, newName)
end

# Accessors
# TODO: Are accessors considered good style instead of accessing directly?
inputs(n::Apply{Variable}) = n.inputs::Vector{Variable}
output(n::Apply{Variable}) = n.output::Variable

pred(n::Apply{Variable}) = n.inputs::Vector{Variable}
succ(n::Apply{Variable}) = [n.output]::Vector{Variable}

#pred(n::Variable) = (isnull(n.owner) ? []::Vector{Apply} : [get(n.owner)])::Vector{Apply}
pred(n::Variable) = isnull(n.owner) ? [] : [get(n.owner)]
succ(n::Variable) = n.clients::Vector{Apply}


immutable Func
    graph::Graph
    outputs::Vector{Variable}
    inputs::Vector{Variable}
    defaults::Dict{Variable, Any}
end

function Func(outputs::Vector{Variable})
    Func(getGraph(outputs), outputs, Vector{Variable}(), Dict{Variable, AbstractArray}())
end

function Func(outputs::Vector{Variable}, inputs::Vector{Variable})
    Func(getGraph(union(outputs, inputs)), outputs, inputs, Dict{Variable, AbstractArray}())
end

function name{T <: Node}(n::T, str::AbstractString)
    n.name = Nullable(str)
    n
end

function apply(op::OpType, inputs::Vector{Variable}, name::AbstractString="")
    #TODO: Is there a way to combine the var and apply creation? Perhaps an inner constructor?
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    var = Variable(TensorVar())
    apply = Apply{Variable}(op, inputs, var, newName)
    var.owner = apply
    for i in inputs
        push!(i.clients, apply)
    end
    var
end

########
# Variable Types
########
# TODO: Decide whether everything is a matrix or not - I think they are
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

type Input <: Tensor
    #shape::Vector{Int}
end

type TensorVar <: Tensor
    #shape::Vector{Int}
end
########
# Operations
########
# TODO: Does this type hierarchy make any sense?

# Create
abstract CreateOp <: OpType
abstract ConstantOp <: CreateOp
abstract RandomOp <: CreateOp  # TODO: Make these!

type Zeros <: ConstantOp end
type Ones <: ConstantOp end
type Fill <: ConstantOp end

type OnesLike <: ConstantOp end

function constant(val::Real, name::AbstractString="")
    Variable(TensorConstant([val], [1,1]), name)
end

function constant(val::AbstractArray, name::AbstractString="")
    Variable(TensorConstant(val, collect(size(val))), name)
end

function fill(shape::Array{Int}, val, name::AbstractString="")
    apply(Fill(), [constant(shape), constant(val)], name)
end

function zeros(shape::Array{Int}, name::AbstractString="")
    fill(shape, 0, name)
end

function ones(shape::Array{Int}, name::AbstractString="")
    fill(shape, 1, name)
end

function ones_like(input::Variable, name::AbstractString="")
    apply(OnesLike(), [input], name)
end

# Specifies an input variable
function input(name::AbstractString="")
    Variable(Input(), name)
end

# Elementwise
# TODO: Is there a better way than instantiating the op as the first argument?
abstract ElementWise <: OpType
# .+ and + are different, just support + and - for now
type Add <: ElementWise end
type Sub <: ElementWise end
type Mul <: ElementWise end
type Div <: ElementWise end

# Matrix math
type MatMul <: OpType end

# Unary operations
abstract UnOp <: OpType
type Neg <: UnOp end

type Transpose <: UnOp end

# TODO: Make += -= etc.
type Assign <: OpType end

########
# Operation creation macros
########

# Each operation needs:
## 1. Type (identifier)
## 2. functions (call or infix, e.g. add and +)
## 3. gradients
## 4. shape inference
## 5. shape inference
## Macros:
### register_ops Type [func, names]
### register_grad Type [grad_body]  # Use ds as variable - similar to ReverseDiff
### register_shape Type [shape body]

macro register_op(typ, op, nargs)
    assert(nargs >= 0)
    args = []
    apply_args = []
    # Var names 'a'...'z'
    for var = 'a':('a' + nargs - 1)
        varsym = symbol(var)
        push!(args, :($(varsym)::Variable))
        push!(apply_args, varsym)
    end
    # TODO: Is there a way to interpolate an expr (like splat) into another expr with $ or similar?
    # For now, use Expr function (for which we can use splat).
    # Actually think it's pretty clear.
    Expr(:function,
         Expr(:call,
              esc(op),
              args...),
         Expr(:call,
              :apply,
              Expr(:call, typ),
              Expr(:vect, apply_args...)))
end


# @register_op(Mul, .*, 2)
# Expands to
# function .*(a::Variable, b::Variable) apply(Mul(), [a, b]) end
@register_op Neg       (-)   1
@register_op Transpose t     1

@register_op MatMul    (*)   2
@register_op Mul       (.*)  2
@register_op Div       (./)  2
@register_op Add       (+)   2
@register_op Sub       (-)   2
@register_op Assign    (.=)  2
@register_op Sub       (-)   2

# TODO: Macro to make it easy to define parts of an operation all together
# e.g. @createOp Add, add, [a, b], [g, g], a + b, [shape inference]
#                type, name, args, gradients, implementation, [shape inference]

######
# Operations implementations
######

op(op::Add, a::AbstractArray, b::AbstractArray) = a + b
op(op::Sub, a::AbstractArray, b::AbstractArray) = a - b
op(op::Mul, a::AbstractArray, b::AbstractArray) = a .* b
op(op::Div, a::AbstractArray, b::AbstractArray) = a ./ b

op(op::MatMul, a::AbstractArray, b::AbstractArray) = a * b

op(op::Neg, a::AbstractArray) = -a

op(op::Transpose, a::AbstractArray) = transpose(a)
op(op::OnesLike, a::AbstractArray) = Base.ones(a)


######
# Gradients
######
# Return vector of variables, where ith is result of gradient wrt input i
function grad(op::Add, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out, grad_out]
end

function grad(op::Sub, inputs::Vector{Variable}, grad_out::Variable)
    res = -grad_out
    [res, res]
end

function grad(op::Mul, inputs::Vector{Variable}, grad_out::Variable)
    [inputs[1] .* grad_out, inputs[2] .* grad_out]
end

function grad(op::Div, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out ./ inputs[1], grad_out ./ inputs[2]]
end

function grad(op::MatMul, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out * t(inputs[2]), t(inputs[1]) * grad_out]
end

function grad(op::Neg, inputs::Vector{Variable}, grad_out::Variable)
    [-grad_out]
end

function grad(op::Transpose, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out]
end

# Autodiff
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
            gradients = grad(node.op, inputs(node), node_to_grad[node.output])
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
    
    node_to_grad
end

## TODO: Build better graph abstraction, avoid repeated code

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

######
# Interpret
######

# Super slow interpret functions
function interpret(f::Func, arguments::Tuple{AbstractArray})
    @assert length(f.inputs) == length(arguments)
    args = Dict{Variable, AbstractArray}()
    for (input, arg) = zip(f.inputs, arguments)
        args[input] = arg
    end
    interpret(f, args)
end

function interpret(f::Func, arguments::Dict{Variable, AbstractArray})
    @assert length(f.inputs) == length(union(keys(arguments), keys(f.defaults)))
    for input_node = f.inputs
        @assert isa(input_node.data, Input)
    end
    frontier = Vector{Apply{Variable}}()
    values = Dict{Variable, AbstractArray{Float}}()

    for node = f.graph.nodes
        if isa(node, Variable)
            if isa(node.data, Input)
                @assert (node in f.inputs) # "Every input node must have a value"
            elseif isnull(node.owner) && !isa(node.data, TensorConstant)
                print(toString(node))
                @assert false && "Every noninput node must have a parent"
            end
        end
    end

    # Set inputs
    for node = f.inputs
        values[node] = arguments[node]
    end

    order = toposort(f.graph)
    for node = order
        if isa(node, Variable)
            if isa(node.data, TensorConstant)
                values[node] = node.data.value
            end
            @assert haskey(values, node)
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
    result = []
    for arg = f.outputs
        @assert haskey(values, arg)
        push!(result, get(values, arg, :impossible))
    end
    result
end

# This is super hacky
# Assumes that first return result is target variable
# SUPER TODO: Fix this
function numeric_grad(f::Func, x::AbstractArray, eps=0.001)
    res1 = interpret(f, (x - eps,))[1]
    res2 = interpret(f, (x + eps,))[1]
    return (res2 - res1) / (2eps * length(x))
end

## TODO: Transform to straight Julia source code
# Time to learn some metaprogramming!

######
# Graph operations
######

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
function getGraph(nodes::Vector{Variable})
    stack = Vector{Node}(nodes)
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
    Graph(seen)
end

##################
# Representation #
##################

# Print connected component of node
function toDot(node::Node)
    G = getGraph([node])
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
        labelLine = string(thisId, " [shape=\"", shape,"\", label=\"", toString(node), "\"];")
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

function toString(node::Variable)
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.data))"
    else
        return "$(typeof(node.data))"
    end
end

function toString(node::Apply)
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.op))"
    else
        return "$(typeof(node.op))"
    end
end

function toString{T <: Node}(nodes::Vector{T})
    c = ", "
    "[$(join(map(toString, nodes), c))]"
end
