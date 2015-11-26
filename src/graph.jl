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
#type Apply{T <: OpType} <: Node
type Apply <: Node
    op::OpType
    inputs::Vector{Node}
    output::Node
    name::Nullable{AbstractString}
end

#type Variable{V <: VarType} <: Node
type Variable <: Node
    owner::Nullable{Apply}
    clients::Vector{Apply}
    data::VarType
    name::Nullable{AbstractString}
end

function Variable(data::VarType, name::AbstractString="")
    #newName = length(name) == 0 ? Nullable() : Nullable(name)
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    Variable(Nullable(), [], data, newName)
end

function apply(op::OpType, inputs::Vector{Variable}, name::AbstractString="")
    newName = length(name) == 0 ? Nullable("$(gensym())") : Nullable(name)
    var = Variable(TensorVar())
    apply = Apply(op, inputs, var, newName)
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
    [grad_out * t(inputs[2]), grad_out * t(inputs[1])]
end

function grad(op::Neg, inputs::Vector{Variable}, grad_out::Variable)
    [-grad_out]
end

function grad(op::Transpose, inputs::Vector{Variable}, grad_out::Variable)
    [grad_out]
end

# Autodiff
function grad(out, wrt)
    
end

######
# Interpret
######

function interpret(outputs::Vector{Variable}, input_nodes::Vector{Input}, arguments::Vector{Array{Float}})
    @assert length(input_nodes) == length(arguments)
    frontier = Vector{Apply}()
    values = Dict{Variable, Array{Float}}()
    graph = Graph(input_nodes)

    for node = graph.nodes
        if typeof(node) == Input && not(node in input_nodes)
            @assert false && "Every input node must have a value"
        elseif isnull(node.owner)
            @assert false && "Every noninput node must have a parent"
        end
    end
end

######
# Graph operations
######

function dfs(seen::Vector{Node})
    # WRITE THIS
end

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

function pred(node::Apply)
    node.inputs
end

function pred(node::Variable)
    isnull(node.owner) ? [] : [get(node.owner)]
end

function succ(node::Apply)
    [node.output]
end

function succ(node::Variable)
    node.clients
end

function label(node::Apply)
    string(typeof(node.op))
end

function label(node::Variable)
    string(typeof(node.data))
end

# Print connected component of node
function toDot(node::Node)
    nodes = connected(node)
    nodeIds = Dict{Node, Int}()
    id = 0
    for node in nodes
        nodeIds[node] = id
        id += 1
    end

    labels = Vector{AbstractString}()
    edges = Vector{AbstractString}()
    for node in nodes
        thisId = nodeIds[node]
        shape = typeof(node) == Apply ? "box" : "ellipse"
        labelLine = string(thisId, " [shape=\"", shape,"\", label=\"", label(node), "\"];")
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
