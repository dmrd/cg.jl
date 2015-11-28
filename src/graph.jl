using Base
importall Base.Operators

typealias Float Float32

abstract Node
abstract VarType
abstract OpType

immutable Graph
    nodes::Set{Node}
end

# OpType - specifies operator (e.g. mul, rand, zeros)
# The lack of mutually recursive types is annoying
#type Apply <: Node
type Apply{T <: Node} <: Node
    op::OpType
    inputs::Vector{T}
    output::T
    name::Nullable{AbstractString}
end


#type Variable{V <: VarType} <: Node
type Variable <: Node
    owner::Nullable{Apply}
    clients::Vector{Apply}
    data::VarType
    name::Nullable{AbstractString}
end

inputs(n::Apply{Variable}) = n.inputs #::Vector{Variable} # Type weirdness
output(n::Apply{Variable}) = n.output::Variable

function name{T <: Node}(n::T, str::AbstractString)
    n.name = Nullable(str)
    n
end

function Variable(data::VarType, name::AbstractString="")
    #newName = length(name) == 0 ? Nullable() : Nullable(name)
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Variable(Nullable(), [], data, newName)
end

function apply(op::OpType, inputs::Vector{Variable}, name::AbstractString="")
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


# type ConstantString <: VarType
#     value::Any
# end

# type ConstantReal <: VarType
#     value::Real
# end

# type ConstantTensor <: Tensor
#     value::AbstractArray
#     shape::Vector{Int}
# end

# Define constant tensors which aren't written out by
# ConstantOp(val, [shape])
# Similar for random

abstract Tensor <: VarType

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



# Create

abstract CreateOp <: OpType
abstract ConstantOp <: CreateOp

type Zeros <: ConstantOp end
type Ones <: ConstantOp end
type Fill <: ConstantOp end

function constant(val::Real, name::AbstractString="")
    Variable(TensorConstant([val], [1,1]), name)
    #Variable{TensorConstant}([val], [1,1])
end

function constant(val::Array, name::AbstractString)
    Variable(TensorConstant(val, collect(size(val))))
end

function fill(shape::Array{Int}, val)
    apply(Fill(), [constant(shape), constant(val)])
end

function zeros(shape::Array{Int})
    fill(shape, 0)
end

function ones(shape::Array{Int})
    fill(shape, 1)
end

function input()
    Variable(Input())
end

# Elementwise
abstract ElementWise <: OpType
type Add <: ElementWise end
type Sub <: ElementWise end
type Mul <: ElementWise end
type Div <: ElementWise end

# Matrix math
type MatAdd <: OpType end
type MatSub <: OpType end
type MatMul <: OpType end

# Unary operations
abstract UnOp <: OpType
type Neg <: UnOp end

type Transpose <: UnOp end

type Assign <: OpType end


-(a::Variable) = apply(Neg(), [a])
Neg(a::Variable) = apply(Neg(), [a])

t(a::Variable) = apply(Transpose(), [a])

+(a::Variable, b::Variable) = apply(MatAdd(), [a, b])
-(a::Variable, b::Variable) = apply(MatSub(), [a, b])
*(a::Variable, b::Variable) = apply(MatMul(), [a, b])

.+(a::Variable, b::Variable) = apply(Add(), [a, b])
.-(a::Variable, b::Variable) = apply(Sub(), [a, b])
.*(a::Variable, b::Variable) = apply(Mul(), [a, b])
./(a::Variable, b::Variable) = apply(Div(), [a, b])

.=(a::Variable, b::Variable) = apply(Assign(), [a, b])

### Gradients
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

function grad(op::MatAdd, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out, grad_out]
end

function grad(op::MatSub, inputs::Vector{Variable}, grad_out::Variable)
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

function grad(op::Assign, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out]
end

# Autodiff
function grad(graph::Graph, out::Variable, wrt::Vector{Variable})
    # Set of nodes on all paths between the set `wrt` and `out`
    # Not
    downstream = influenced_by(wrt, true)
    upstream = influenced_by([out], false)
    on_path = intersect(upstream, downstream)

    toposorted = toposort(graph)
    node_to_grad = Dict{Variable, Variable}(out => out)
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
# Operations
######


######
# Interpret
######


function interpret(graph::Graph, outputs::Vector{Variable}, input_nodes::Vector{Input}, arguments::Vector{Array{Float}})
    @assert length(input_nodes) == length(arguments)
    frontier = Vector{Apply{Variable}}()
    values = Dict{Variable, Array{Float}}()
    graph = Graph(input_nodes)

    for node = graph.nodes
        if typeof(node) == Input && not(node in input_nodes)
            @assert false && "Every input node must have a value"
        elseif isnull(node.owner)
            @assert false && "Every noninput node must have a parent"
        end
    end

    order = toposorted(graph)
    for node = order
        if isa(node, Variable)
            
        elseif isa(node, Apply)

        end
    end
end

## TODO: Transform to straight Julia source code

######
# Graph operations
######

function dfs(seen::Vector{Node})
    # WRITE THIS
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

pred(node::Apply) = inputs(node)
succ(node::Apply) = [node.output]

# function pred(node::Apply{T})
#     node.inputs
# end
# function succ(node::Apply{T})
#     [node.output]
# end

function pred(node::Variable)
    isnull(node.owner) ? [] : [get(node.owner)]
end


function succ(node::Variable)
    node.clients
end

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

# for op = (:+, :-, :*, :/)
#   @eval ($op)(a::Variable, b::Variable) = opn{Tensor}(BinOp(op), a, b)
# end


## Functions CGT implements
