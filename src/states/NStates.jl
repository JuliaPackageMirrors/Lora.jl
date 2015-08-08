### Abstract variable nstates

abstract VariableNState{F<:VariateForm, N<:Number}

abstract GenericVariableNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

abstract ParameterNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{GenericVariableNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterNState{F, N}}) = N

### Generic variable nstate subtypes

## UnivariateGenericVariableNState

type UnivariateGenericVariableNState{N<:Number} <: GenericVariableNState{Univariate, N}
  value::Vector{N}
  n::Int
end

UnivariateGenericVariableNState{N<:Number}(value::Vector{N}) = UnivariateGenericVariableNState{N}(value, length(value))

Base.eltype{N<:Number}(::Type{UnivariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::UnivariateGenericVariableNState{N}) = N

Base.convert(::Type{UnivariateGenericVariableNState}, state::UnivariateGenericVariableState) =
  UnivariateGenericVariableNState(eltype(state)[state.value], 1)

save!(nstate::UnivariateGenericVariableNState, state::UnivariateGenericVariableState, i::Int) =
  (nstate.value[i] = state.value)

## MultivariateGenericVariableNState

type MultivariateGenericVariableNState{N<:Number} <: GenericVariableNState{Multivariate, N}
  value::Matrix{N}
  size::Int
  n::Int
end

MultivariateGenericVariableNState{N<:Number}(value::Matrix{N}) =
  MultivariateGenericVariableNState{N}(value, size(value, 2), size(value, 1))

Base.eltype{N<:Number}(::Type{MultivariateGenericVariableNState{N}}) = N
Base.eltype{N<:Number}(s::MultivariateGenericVariableNState{N}) = N

Base.convert(::Type{MultivariateGenericVariableNState}, state::MultivariateGenericVariableState) =
  MultivariateGenericVariableNState(transpose(state.value), length(state.value), 1)
