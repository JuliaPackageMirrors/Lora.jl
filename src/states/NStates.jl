### Abstract variable NStates

abstract VariableNState{F<:VariateForm, N<:Number}

abstract GenericVariableNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

abstract ParameterNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

typealias MCChain ParameterNState

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{GenericVariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterNState{F, N}}) = N

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

save!(nstate::UnivariateGenericVariableNState, state::UnivariateGenericVariableState, i::Int) =
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

save!(nstate::MultivariateGenericVariableNState, state::MultivariateGenericVariableState, i::Int) =
  (nstate.value[1:state.size, i] = state.value)

## MatrixvariateGenericVariableNState

type MatrixvariateGenericVariableNState{N<:Number} <: GenericVariableNState{Matrixvariate, N}
  value::Array{N, 3}
  size::Tuple
  n::Int
end

MatrixvariateGenericVariableNState{N<:Number}(value::Array{N, 3}) =
  MatrixvariateGenericVariableNState{N}(value, (size(value, 1), size(value, 2)), size(value, 3))

MatrixvariateGenericVariableNState{N<:Number}(::Type{N}, size::Tuple=(0, 0), n::Int=0) =
  MatrixvariateGenericVariableNState{N}(Array(N, size..., n), size, n)

Base.eltype{N<:Number}(::Type{MatrixvariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MatrixvariateGenericVariableNState{N}) = N

save!(nstate::MatrixvariateGenericVariableNState, state::MatrixvariateGenericVariableState, i::Int) =
  (nstate.value[1:state.size[1], 1:state.size[2], i] = state.value)
