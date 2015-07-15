### Abstract variable states

abstract VariableState{F<:VariateForm, N<:Number}

abstract GenericVariableState{F<:VariateForm, N<:Number} <: VariableState{F, N}

abstract ParameterState{F<:VariateForm, N<:Number} <: VariableState{F, N}

### Generic variable state subtypes

## UnivariateGenericVariableState

type UnivariateGenericVariableState{N<:Number} <: GenericVariableState{Univariate, N}
  value::N
end

## MultivariateGenericVariableState

type MultivariateGenericVariableState{N<:Number} <: GenericVariableState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateGenericVariableState{N<:Number}(value::Vector{N}) =
  MultivariateGenericVariableState{N}(value, length(value))

## MatrixvariateGenericVariableState

type MatrixvariateGenericVariableState{N<:Number} <: GenericVariableState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple
end

MatrixvariateGenericVariableState{N<:Number}(value::Matrix{N}) =
  MatrixvariateGenericVariableState{N}(value, size(value))

### Parameter state subtypes

## ContinuousUnivariateParameterState

type ContinuousUnivariateParameterState{N<:FloatingPoint} <: ParameterState{Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorlogtarget::N
  dtensorlogtarget::N
  diagnostics::Dict
end

function ContinuousUnivariateParameterState{N<:FloatingPoint}(value::N)
  v = convert(N, NaN)
  ContinuousUnivariateParameterState{N}(value, v, v, v, v, v, v, v, v, Dict())
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(::Type{N}) = ContinuousUnivariateParameterState(convert(N, NaN))

ContinuousUnivariateParameterState{N<:FloatingPoint}(;
  value::N = convert(N, NaN),
  loglikelihood::N=convert(N, NaN),
  logprior::N=convert(N, NaN),
  logtarget::N=convert(N, NaN),
  gradloglikelihood::N=convert(N, NaN),
  gradlogprior::N=convert(N, NaN),
  gradlogtarget::N=convert(N, NaN),
  tensorlogtarget::N=convert(N, NaN),
  dtensorlogtarget::N=convert(N, NaN),
  diagnostics::Dict=Dict()
) =
  ContinuousUnivariateParameterState{N}(
    value,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorlogtarget,
    dtensorlogtarget,
    diagnostics
  )

## ContinuousMultivariateParameterState

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ParameterState{Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorlogtarget::Matrix{N}
  dtensorlogtarget::Array{N, 3}
  size::Int
  diagnostics::Dict
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(value::Vector{N}, monitor::Vector{Bool}=fill(false, 5))
  size = length(value)

  l = Array(Int, 5)
  for i in 1:5
    l[i] = (monitor[i] == false ? zero(Int) : size)
  end

  v = convert(N, NaN)

  ContinuousMultivariateParameterState{N}(
    value,
    v,
    v,
    v,
    Array(N, l[1]),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4], l[4]),
    Array(N, l[5], l[5], l[5]),
    size,
    Dict()
  )
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(value::Vector{N}, monitor::Dict{Symbol, Bool})
  fields = (:gradloglikelihood, :gradlogprior, :gradlogtarget, :tensorlogtarget, :dtensorlogtarget)
  ContinuousMultivariateParameterState(
    value, 
    Bool[haskey(monitor, fields[i]) ? monitor[fields[i]] : false for i in 1:5]
  )
end

ContinuousMultivariateParameterState{N<:FloatingPoint}(::Type{N}, size::Int=0, monitor::Vector{Bool}=fill(false, 5)) =
  ContinuousMultivariateParameterState(Array(N, size), monitor)

ContinuousMultivariateParameterState{N<:FloatingPoint}(::Type{N}, size::Int, monitor::Dict{Symbol, Bool}) =
  ContinuousMultivariateParameterState(Array(N, size), monitor)

ContinuousMultivariateParameterState{N<:FloatingPoint}(;
  value::Vector{N}=Array(N, zero(Int)),
  loglikelihood::N=convert(N, NaN),
  logprior::N=convert(N, NaN),
  logtarget::N=convert(N, NaN),
  gradloglikelihood::Vector{N}=Array(N, zero(Int)),
  gradlogprior::Vector{N}=Array(N, zero(Int)),
  gradlogtarget::Vector{N}=Array(N, zero(Int)),
  tensorlogtarget::Matrix{N}=Array(N, zero(Int), zero(Int)),
  dtensorlogtarget::Array{N, 3}=Array(N, zero(Int), zero(Int), zero(Int)),
  size::Int=length(value),
  diagnostics::Dict=Dict()
) =
  ContinuousMultivariateParameterState{N}(
    value,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorlogtarget,
    dtensorlogtarget,
    size,
    diagnostics
  )
