using Base
importall Base.Operators

typealias Float Float32

abstract Node
abstract VarType
abstract OpType

# OpType - specifies operator (e.g. mul, rand, zeros)
#type Apply{T <: OpType} <: Node
type Apply <: Node
    op::OpType
    inputs::Vector{Node}
    output::Node
end

#type Variable{V <: VarType} <: Node
type Variable <: Node
    owner::Nullable{Apply}
    clients::Vector{Apply}
    data::VarType
end

function Variable(data::VarType)
    Variable(Nullable(), [], data)
end

function apply(op::OpType, inputs::Vector{Variable})
    var = Variable(TensorVar())
    apply = Apply(op, inputs, var)
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
    shape::Vector{Int}
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
type Constant <: ConstantOp end

function constant(val::Real)
    Variable(TensorConstant([val], [1,1]))
    #Variable{TensorConstant}([val], [1,1])
end

function constant(val::Array)
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

function input(shape)
    Variable(Input(shape))
end

# Matrix math
type MatMul <: OpType end
type MatAdd <: OpType end
type MatSub <: OpType end

# Elementwise
abstract ElementWise <: OpType
type Add <: ElementWise end
type Sub <: ElementWise end
type Mul <: ElementWise end
type Div <: ElementWise end
type Mod <: ElementWise end

# Unary operations
abstract UnOp <: OpType
type Abs <: UnOp end
type Neg <: UnOp end

abs(a::Variable) = apply(Abs(), [a])
-(a::Variable) = apply(Neg(), [a])

+(a::Variable, b::Variable) = apply(MatAdd(), [a, b])
-(a::Variable, b::Variable) = apply(MatSub(), [a, b])
*(a::Variable, b::Variable) = apply(MatMul(), [a, b])

.+(a::Variable, b::Variable) = apply(Add(), [a, b])
.-(a::Variable, b::Variable) = apply(Sub(), [a, b])
.*(a::Variable, b::Variable) = apply(Mul(), [a, b])
./(a::Variable, b::Variable) = apply(Div(), [a, b])
