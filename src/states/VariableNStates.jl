### Abstract variable NStates

abstract VariableNState{F<:VariateForm, N<:Number}

abstract GenericVariableNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{GenericVariableNState{F, N}}) = N

add_dimension(n::Number) = eltype(n)[n]
add_dimension(a::Array, sa::Tuple=size(a)) = reshape(a, sa..., 1)

### Generic variable NState subtypes

## UnivariateGenericVariableNState

type UnivariateGenericVariableNState{N<:Number} <: GenericVariableNState{Univariate, N}
  value::Vector{N}
  n::Int
end

UnivariateGenericVariableNState{N<:Number}(value::Vector{N}) = UnivariateGenericVariableNState{N}(value, length(value))

UnivariateGenericVariableNState{N<:Number}(::Type{N}, n::Int=0) = UnivariateGenericVariableNState{N}(Array(N, n), n)

Base.eltype{N<:Number}(::Type{UnivariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::UnivariateGenericVariableNState{N}) = N

Base.copy!(nstate::UnivariateGenericVariableNState, state::UnivariateGenericVariableState, i::Int) =
  (nstate.value[i] = state.value)

## MultivariateGenericVariableNState

type MultivariateGenericVariableNState{N<:Number} <: GenericVariableNState{Multivariate, N}
  value::Matrix{N}
  size::Int
  n::Int
end

MultivariateGenericVariableNState{N<:Number}(value::Matrix{N}) =
  MultivariateGenericVariableNState{N}(value, size(value)...)

MultivariateGenericVariableNState{N<:Number}(::Type{N}, size::Int=0, n::Int=0) =
  MultivariateGenericVariableNState{N}(Array(N, size, n), size, n)

Base.eltype{N<:Number}(::Type{MultivariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MultivariateGenericVariableNState{N}) = N

Base.copy!(nstate::MultivariateGenericVariableNState, state::MultivariateGenericVariableState, i::Int) =
  (nstate.value[1+(i-1)*state.size:i*state.size] = state.value)

## MatrixvariateGenericVariableNState

type MatrixvariateGenericVariableNState{N<:Number} <: GenericVariableNState{Matrixvariate, N}
  value::Array{N, 3}
  size::Tuple{Int, Int}
  n::Int
end

MatrixvariateGenericVariableNState{N<:Number}(value::Array{N, 3}) =
  MatrixvariateGenericVariableNState{N}(value, (size(value, 1), size(value, 2)), size(value, 3))

MatrixvariateGenericVariableNState{N<:Number}(::Type{N}, size::Tuple=(0, 0), n::Int=0) =
  MatrixvariateGenericVariableNState{N}(Array(N, size..., n), size, n)

Base.eltype{N<:Number}(::Type{MatrixvariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MatrixvariateGenericVariableNState{N}) = N

Base.copy!(
  nstate::MatrixvariateGenericVariableNState,
  state::MatrixvariateGenericVariableState,
  i::Int,
  statelen::Int=prod(state.size)
) =
  (nstate.value[1+(i-1)*statelen:i*statelen] = state.value)
