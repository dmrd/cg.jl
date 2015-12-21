####
# Operations
####

typealias Shape Vector{Int}
# TODO: Shape inference
#### TODO: Broadcasting (see ?broadcast)
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
    # Think it's pretty clear for now
    args, apply_args = gen_args(narg, Node)
    Expr(:function,
         Expr(:call,
              esc(op),
              args...,
              Expr(:kw,
                   Expr(:(::),
                        :name,
                        :AbstractString),
                   ""
                   )
              ),
         Expr(:call,
              :Node,
              Expr(:call, typ),
              Expr(:vect, apply_args...),
              :name))
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
function call(op::Mul, a::TensorValue, b::TensorValue, c::TensorValue)
    a + b + c
end
"""
macro register_impl(typ, narg, impl)
    args, _ = gen_args(narg, TensorValue)
    Expr(:function,
         Expr(:call,
              esc(:call),
              :(op::$typ),
              args...),
         impl)
end

#########################
# Operation Definitions #
#########################
# TODO: Define all parts of an operation together or keep similar parts grouped?
# e.g. @createOp Add, add, [a, b], [g, g], a + b, [shape inference]
#                type, name, args, gradients, implementation, [shape inference]
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
function call(op::Noop, a::TensorValue...)
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
