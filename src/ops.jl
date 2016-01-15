####
# Shape
####
# TODO: Shape inference
typealias Shape Vector{Int}

####
# Operations
####

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
immutable Print <: OpType end
immutable Noop <: OpType end

immutable Fill <: ConstantOp end
immutable FillLike <: ConstantOp end

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

type RandN <: RandomOp end
type Copy <: ElementWise end

type Dot <: OpType end  # Matrix multiply
type Transpose <: OpType end
type Sum <: OpType end
type Dim <: OpType end
type Assign <: OpType end
type InPlaceAdd <: OpType end

type Mean <: OpType end
type Sum <: OpType end
type Maximum <: OpType end
type RepeatTo <: OpType end

# These are mainly for testing
type GetIndex <: OpType end
type SetIndex <: OpType end

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
function .*(a::Node, b::Node) Node(Mul(), [a, b]) end
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
function grad(op::Mul, out::Node, ds::Node, a::Node, b::Node)
    [a .* ds, b .* grad_out]
end
TODO: Make this take inputs[], outputs[], gradients[] explicitly as lists
"""
macro register_grad(typ, grads...)
    args, _ = gen_args(length(grads), Node)
    Expr(:function,
         Expr(:call,
              esc(:grad),
              :(op::$typ),
              :(out::Node),
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

"""
@register_shape Mul 3 (a + b)
Expands to
function shape(op::Mul, a::Shape, b::Shape)
    # Check compatability
    ...
    # Return new result
    a + b
end
"""
macro register_shape(typ, narg, impl)
    args, _ = gen_args(narg, Shape)
    Expr(:function,
         Expr(:call,
              esc(:shape),
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
@register_op Print       print        1
@register_op Dim         dim          1
@register_op Fill        fill         2
@register_op FillLike    fill_like    2
@register_op Copy        copy         1

@register_op Dot         dot          2
@register_op Transpose   t            1
@register_op Assign      (.=)         2
@register_op InPlaceAdd  plusequals   2  # += doesn't work


@register_op Sum         sum          1
@register_op Sum         sum          2 # Arg 2 = axis
@register_op Mean        mean         1
@register_op Mean        mean         2

#@register_op Relu        relu         1

#@register_op SoftMax     softmax      1

@register_op RandN       randn        1

@register_op Maximum     maximum      1
@register_op Maximum     maximum      2

@register_op GetIndex    getindex     2
@register_op SetIndex    setindex     3

@register_op RepeatTo    repeatto     2 # input array, target shp - expand singleton dims

################################

@register_impl Print        1   (println(a); a)
@register_impl Constant     0   op.value
# a = scalar, b = 1d array specifying dims
@register_impl Fill         2   Base.fill(a, b...)
# a = scalar, b = array shape to copy
@register_impl FillLike     2   (dim = size(b); length(dim) == 0 ? a : fill(a, dim...))

@register_impl Dim          1   collect(Int, size(a))

@register_impl Copy         1   a

@register_impl Dot          2   a * b
@register_impl Transpose    1   transpose(a)
# TODO: Run without returning anything.  Explicit NoValue
@register_impl InPlaceAdd   2   (for i in 1:length(a); a[i] += b[i] end; 0)

@register_impl Sum          1   Base.sum(a)
@register_impl Sum          2   Base.sum(a, b)
@register_impl Mean         1   Base.mean(a)
@register_impl Mean         2   Base.mean(a, b)

@register_impl RandN         1   Base.randn(a...)

@register_impl Maximum       1   maximum(a)
@register_impl Maximum       2   maximum(a, b)

@register_impl GetIndex      2   (a[b])
@register_impl SetIndex      3   (t = copy(a); t[b] = c; t)


function call(op::RepeatTo, a::Real, dim::Array)
    fill(a, dim...)
end

# Repeats singleton dimensions of the given array to given dimensions
function call(op::RepeatTo, a::Array, dim::Array)
    cdim = size(a)
    ncurdim = length(cdim)
    @assert length(size(dim)) == 1  # Is a vector
    @assert length(dim) >= length(cdim)  # target has at least as many dimensions
    repcount = zeros(dim)
    for i = 1:length(dim)
        if i > ncurdim
            # Expand all implicitly 1 dimensions
            repcount[i] = dim[i]
        elseif cdim[i] == 1
            # Expand all singleton dims
            repcount[i] = dim[i]
        elseif cdim[i] == dim[i]
            # Leave rest the same
            repcount[i] = 1
        else
            # All all nonsingleton dimensions must be the same
            @assert false
        end
    end
    repeat(a, inner=repcount)
end

################################

@register_grad Print ds
@register_grad Dot broadcast("*", ds, t(b)) broadcast("*", t(a), ds)
@register_grad Transpose ds

@register_grad Sum repeatto(ds, dim(a))
@register_grad Sum repeatto(ds, dim(a)) (cg.constant(0)) # TODO: make axis nondiff

# This is wrong for edge cases
@register_grad Maximum  (eq(a, out) * ds)
@register_grad Maximum  broadcast(Mul(), broadcast(Eq(), a, out), ds) (b)  # 2nd one should be undefined

# Incredibly inefficient, but mostly for testing
# TODO: 0s Replace with nondiff
@register_grad GetIndex  (t = fill(cg.constant(0.0), dim(a)); setindex(t, b, ds)) (cg.constant(0))
@register_grad SetIndex  (setindex(ds, b, cg.constant(0.0))) (cg.constant(0)) (getindex(ds, b))

# TODO How to treate OnesLike etc. in gradient computations?
# TODO: Add actual GradUndefined
## TODO: May want to start grouping together like this

type Softmax <: OpType end
#@register_op Softmax softmax 1
#@register_impl SoftMax      1   (m = maximum(a, 1); subbed = a .- m; exped = exp(subbed); exped ./ sum(exped, 1))
#@register_grad SoftMax      1   (a = maximum())




###########
# Scalars #
###########
abstract ScalarOp <: OpType

type Add <: ScalarOp end
type Sub <: ScalarOp end
type Mul <: ScalarOp end
type Div <: ScalarOp end
type Pow <: ScalarOp end

type Neg <: ScalarOp end
type Sign <: ScalarOp end
type Exp <: ScalarOp end
type Log <: ScalarOp end
type Sin <: ScalarOp end
type Cos <: ScalarOp end
type Abs <: ScalarOp end

type Max <: ScalarOp end
type Min <: ScalarOp end

type Eq <: ScalarOp end
type Neq <: ScalarOp end
type Le <: ScalarOp end
type Leq <: ScalarOp end
type Ge <: ScalarOp end
type Geq <: ScalarOp end

type Sigmoid <: ScalarOp end

@register_op Add     (+)          2
@register_op Sub     (-)          2
@register_op Mul     (*)          2
@register_op Div     (/)          2
@register_op Pow     (^)          2

@register_op Neg         (-)          1
@register_op Sign        sign         1
@register_op Exp         exp          1
@register_op Log         log          1
@register_op Sin         sin          1
@register_op Cos         cos          1
@register_op Abs         abs          1

@register_op Max     max          2
@register_op Min     min          2

# Would like to use == etc., but spell out for now
@register_op Eq      (eq)         2
@register_op Neq     (neq)        2
@register_op Le      (le)         2
@register_op Leq     (leq)        2
@register_op Ge      (ge)         2
@register_op Geq     (geq)        2

@register_op Sigmoid     sigmoid      1

# Basically reimplementing Base here
function elwise(op::Function, a::Real, b::Real)
    op(a, b)
end

# TODO Similar isn't really right - should have proper type
function elwise(op::Function, a::Real, b::Array)
    out = similar(b)
    for i = eachindex(b)
        @inbounds out[i] = op(a, b[i])
    end
    reshape(out, size(b))
end
function elwise(op::Function, a::Array, b::Real)
    out = similar(a)
    for i = eachindex(a)
        @inbounds out[i] = op(a[i], b)
    end
    reshape(out, size(a))
end
function elwise(op::Function, a::Array, b::Array)
    @assert size(a) == size(b)
    out = similar(a)
    for i = eachindex(a)
        @inbounds out[i] = op(a[i], b[i])
    end
    reshape(out, size(a))
end

call(op::Add,  a::TensorValue, b::TensorValue)  = elwise(+, a, b)
call(op::Sub,  a::TensorValue, b::TensorValue)  = elwise(-, a, b)
call(op::Mul,  a::TensorValue, b::TensorValue)  = elwise(*, a, b)
call(op::Div,  a::TensorValue, b::TensorValue)  = elwise(/, a, b)
call(op::Pow,  a::TensorValue, b::TensorValue)  = elwise(^, a, b)

# These are autobroadcast
call(op::Neg,  a::TensorValue)           = -a
call(op::Sign, a::TensorValue)           = sign(a)
call(op::Exp,  a::TensorValue)           = exp(a)
call(op::Log,  a::TensorValue)           = log(a)
call(op::Sin,  a::TensorValue)           = sin(a)
call(op::Cos,  a::TensorValue)           = cos(a)
call(op::Abs,  a::TensorValue)           = abs(a)

call(op::Max,  a::TensorValue, b::TensorValue)  = elwise(max, a, b)
call(op::Min,  a::TensorValue, b::TensorValue)  = elwise(min, a, b)

call(op::Eq,   a::TensorValue, b::TensorValue)  = elwise(==, a, b)
call(op::Neq,  a::TensorValue, b::TensorValue)  = elwise(!=, a, b)
call(op::Le,   a::TensorValue, b::TensorValue)  = elwise(<, a, b)
call(op::Leq,  a::TensorValue, b::TensorValue)  = elwise(<=, a, b)
call(op::Ge,   a::TensorValue, b::TensorValue)  = elwise(>, a, b)
call(op::Geq,  a::TensorValue, b::TensorValue)  = elwise(>=, a, b)

call(op::Sigmoid, a::TensorValue) = (1.0 ./ (1.0 + exp(-a))) 

# TODO: This is absolutely definitely not a good long term solution
@register_grad Add bg(ds, a) bg(ds, b)
@register_grad Sub bg(ds, a) bg(-ds, b)
@register_grad Mul bg(b * ds, a) bg(a * ds, b)
@register_grad Div bg(ds / b, a) bg(-(ds * a) / (b * b), b)
@register_grad Pow bg(ds * b * a ^ (b - cg.constant(1.0)), a) bg(ds * log(a) * (a ^ b), b)
# TODO: make the constants the proper type

@register_grad Sign fill_like(cg.constant(0), a)
@register_grad Neg (-ds)
@register_grad Exp (exp(a) * ds)
@register_grad Log (ds / a)
@register_grad Sin (cos(a) * ds)
@register_grad Cos (-sin(a) * ds)
@register_grad Abs (sign(a) * ds)

@register_grad Max bg(eq(out, a) * ds, a) bg(eq(out, b) * ds, b)
@register_grad Min bg(eq(out, a) * ds, a) bg(eq(out, b) * ds, b)

@register_grad Sigmoid (sigmoid(a) * (cg.constant(1.0) - sigmoid(a)) * ds)

################
# Broadcasting #
################
# This is used for arrays that are different sizes.
# Broadcast singleton dimensions to the same size
# TODO decision: Should all broadcasts be explicit?
# Handling broadcasting is surprisingly tricky, especially grads (compile time v. runtime)

# Some way to specialize {T <: ScalarOp}?
type Broadcast <: OpType
    op::ScalarOp
end
type BroadcastGrad <: OpType
    op::ScalarOp
end

function broadcastop(op::ScalarOp, a::Node, b::Node)
    Node(Broadcast(op), [a, b])
end

function broadcastgrad(op::ScalarOp, a::Node, b::Node)
    Node(Broadcast(op), [a, b])
end

function broadcast(name::AbstractString, a::Node, b::Node)
    if name== "+"
        op = Add()
    elseif name == "-"
        op = Sub()
    elseif name == "*"
        op = Mul()
    elseif name == "/"
        op = Div()
    elseif name == "^"
        op = Pow()
    end
    broadcastop(op, a, b)
end

@register_impl Broadcast 2 broadcast(op.op, a, b)
@register_grad Broadcast broadcastgrad(ds, a) broadcastgrad(ds, b)
# ^ This is wrong - doesn't actually use the op.grad

# TODO: This feels like a hack even if it works
# In place until shape inference exists?
# args: [out, arg to go back to]
# Alternative is some explicit copying
@register_op BroadcastGrad broadcastgrad 2
bg = broadcastgrad

# What axes to sum to get from cur_dim to target
function get_sum_dims(curdim::Vector{Int}, targetdim::Vector{Int})
    dims = Vector{Int64}()
    nt = length(targetdim)
    nc = length(curdim)
    @assert nt <= nc
    for dim = 1:nc
        if dim > nt && curdim[dim] > 1
            # Sum along all implicitly 1 axes of target
            push!(dims, dim)
        elseif curdim[dim] != targetdim[dim] == 1
            push!(dims, dim)
        end
    end
    dims
end

# Sum axes of "from" to match dimensions of "to"
function call(op::BroadcastGrad, from::Real, to::Real)
    from
end

function call(op::BroadcastGrad, from::Real, to::Array)
    fill(from, size(to)...)
end

function call(op::BroadcastGrad, from::Array, to::Real)
    sum(from)
end

function call(op::BroadcastGrad, from::Array, to::Array)
    fshape = size(from)
    tshape = size(to)
    if fshape == tshape
        return from
    end
    dims = get_sum_dims(collect(fshape), collect(tshape))
    return sum(from, dims)
end


#TODO question: Is {T <: ScalarOp}(op::T) more efficient than op::ScalarOp?  Look at codegen
#TODO Just calling sin(x) is faster than broadcast(sin, x) by many times.  Do this instead for most ops

###############
# Complex ops #
###############

function crossentropy(label::Node, prediction::Node)
    result = -sum(label * log(prediction), cg.constant(1))
    group_between([label, prediction], [result], string(gensym(:crossentropy)), include_in=false)
    result
end

function softmax(node::Node)
    # max = maximum(node, constant(1))  # Maximum columnwise
    # exped = exp(node - max)
    node = print(node)
    exped = exp(node)
    exped = print(exped)
    summed = sum(exped, constant(1))
    summed = print(summed)
    div = broadcast("/", exped, summed)
    div = print(div)
    group_between([node], [div], string(gensym(:softmax)), include_in=false)
    div
end

# TODO: Add some numeric stability optimizations so we don't need this
function softmax_crossentropy(label::Node, unnorm_prediction::Node)
    max = maximum(unnorm_prediction)
    exped = exp(unnorm_prediction - max)
    summed = sum(exped, constant(1))
    lg = unnorm_prediction - log(summed)
    result = -sum(label * lg)
    group_between([label, unnorm_prediction], [result], string(gensym(:softmax_crossentropy)), include_in=false)
    result
end

function mean_squared_error(a::Node, b::Node)
    diff = a - b
    sq = diff .* diff
    mean(sq)
end


################
# Optimization #
################

# Create an optimize op and return 
function sgd_optimizer(loss::Node, variables::Vector{Node}, step_size::Node)
    gradients = grad(loss, variables)
    step_sizes = map(grad -> step_size * grad, gradients)
    updates = map(vargrad -> plusequals(vargrad[1], (-step_size * vargrad[2])), zip(variables, gradients))
    noop(updates...)
end
