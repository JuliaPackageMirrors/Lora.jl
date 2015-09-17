### Constants associated with states and NStates

const main_state_field_names = (
  :value,
  :loglikelihood,
  :logprior,
  :logtarget,
  :gradloglikelihood,
  :gradlogprior,
  :gradlogtarget,
  :tensorloglikelihood,
  :tensorlogprior,
  :tensorlogtarget,
  :dtensorloglikelihood,
  :dtensorlogprior,
  :dtensorlogtarget,
  :diagnostics
)

### Abstract variable states

abstract VariableState{F<:VariateForm, N<:Number}

abstract GenericVariableState{F<:VariateForm, N<:Number} <: VariableState{F, N}

abstract ParameterState{F<:VariateForm, N<:Number} <: VariableState{F, N}

abstract ContinuousParameterState{F<:VariateForm, N<:FloatingPoint} <: ParameterState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{GenericVariableState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:FloatingPoint}(::Type{ContinuousParameterState{F, N}}) = N

### Generic variable state subtypes

## UnivariateGenericVariableState

type UnivariateGenericVariableState{N<:Number} <: GenericVariableState{Univariate, N}
  value::N
end

Base.eltype{N<:Number}(::Type{UnivariateGenericVariableState{N}}) = N
Base.eltype{N<:Number}(s::UnivariateGenericVariableState{N}) = N

## MultivariateGenericVariableState

type MultivariateGenericVariableState{N<:Number} <: GenericVariableState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateGenericVariableState{N<:Number}(value::Vector{N}) =
  MultivariateGenericVariableState{N}(value, length(value))

MultivariateGenericVariableState{N<:Number}(::Type{N}, size::Int=0) =
  MultivariateGenericVariableState{N}(Array(N, size), size)

Base.eltype{N<:Number}(::Type{MultivariateGenericVariableState{N}}) = N
Base.eltype{N<:Number}(s::MultivariateGenericVariableState{N}) = N

## MatrixvariateGenericVariableState

type MatrixvariateGenericVariableState{N<:Number} <: GenericVariableState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple{Int, Int}
end

MatrixvariateGenericVariableState{N<:Number}(value::Matrix{N}) =
  MatrixvariateGenericVariableState{N}(value, size(value))

MatrixvariateGenericVariableState{N<:Number}(::Type{N}, size::Tuple=(0, 0)) =
  MatrixvariateGenericVariableState{N}(Array(N, size...), size)

Base.eltype{N<:Number}(::Type{MatrixvariateGenericVariableState{N}}) = N
Base.eltype{N<:Number}(s::MatrixvariateGenericVariableState{N}) = N
