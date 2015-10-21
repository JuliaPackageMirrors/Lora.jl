### Abstract variable NStates

abstract VariableNState{F<:VariateForm, N<:Number}

abstract BasicVariableNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{BasicVariableNState{F, N}}) = N

add_dimension(n::Number) = eltype(n)[n]
add_dimension(a::Array, sa::Tuple=size(a)) = reshape(a, sa..., 1)

### Basic variable NState subtypes

## UnivariateBasicVariableNState

type UnivariateBasicVariableNState{N<:Number} <: BasicVariableNState{Univariate, N}
  value::Vector{N}
  n::Int
end

UnivariateBasicVariableNState{N<:Number}(value::Vector{N}) = UnivariateBasicVariableNState{N}(value, length(value))

UnivariateBasicVariableNState{N<:Number}(::Type{N}, n::Int=0) = UnivariateBasicVariableNState{N}(Array(N, n), n)

Base.eltype{N<:Number}(::Type{UnivariateBasicVariableNState{N}}) = N
Base.eltype{N<:Number}(s::UnivariateBasicVariableNState{N}) = N

Base.copy!(nstate::UnivariateBasicVariableNState, state::UnivariateBasicVariableState, i::Int) =
  (nstate.value[i] = state.value)

## MultivariateBasicVariableNState

type MultivariateBasicVariableNState{N<:Number} <: BasicVariableNState{Multivariate, N}
  value::Matrix{N}
  size::Int
  n::Int
end

MultivariateBasicVariableNState{N<:Number}(value::Matrix{N}) = MultivariateBasicVariableNState{N}(value, size(value)...)

MultivariateBasicVariableNState{N<:Number}(::Type{N}, size::Int=0, n::Int=0) =
  MultivariateBasicVariableNState{N}(Array(N, size, n), size, n)

Base.eltype{N<:Number}(::Type{MultivariateBasicVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MultivariateBasicVariableNState{N}) = N

Base.copy!(nstate::MultivariateBasicVariableNState, state::MultivariateBasicVariableState, i::Int) =
  (nstate.value[1+(i-1)*state.size:i*state.size] = state.value)

## MatrixvariateBasicVariableNState

type MatrixvariateBasicVariableNState{N<:Number} <: BasicVariableNState{Matrixvariate, N}
  value::Array{N, 3}
  size::Tuple{Int, Int}
  n::Int
end

MatrixvariateBasicVariableNState{N<:Number}(value::Array{N, 3}) =
  MatrixvariateBasicVariableNState{N}(value, (size(value, 1), size(value, 2)), size(value, 3))

MatrixvariateBasicVariableNState{N<:Number}(::Type{N}, size::Tuple=(0, 0), n::Int=0) =
  MatrixvariateBasicVariableNState{N}(Array(N, size..., n), size, n)

Base.eltype{N<:Number}(::Type{MatrixvariateBasicVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MatrixvariateBasicVariableNState{N}) = N

Base.copy!(
  nstate::MatrixvariateBasicVariableNState,
  state::MatrixvariateBasicVariableState,
  i::Int,
  statelen::Int=prod(state.size)
) =
  (nstate.value[1+(i-1)*statelen:i*statelen] = state.value)
