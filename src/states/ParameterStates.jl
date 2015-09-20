### Abstract parameter states

abstract ParameterState{F<:VariateForm, N<:Number} <: VariableState{F, N}

abstract ContinuousParameterState{F<:VariateForm, N<:FloatingPoint} <: ParameterState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:FloatingPoint}(::Type{ContinuousParameterState{F, N}}) = N

### Constants associated with continuous parameter states and NStates

const main_cpstate_fields = (
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
  :dtensorlogtarget
)

### Parameter state subtypes

## ContinuousUnivariateParameterState

type ContinuousUnivariateParameterState{N<:FloatingPoint} <: ContinuousParameterState{Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorloglikelihood::N
  tensorlogprior::N
  tensorlogtarget::N
  dtensorloglikelihood::N
  dtensorlogprior::N
  dtensorlogtarget::N
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Vector
end

function ContinuousUnivariateParameterState{N<:FloatingPoint}(
  value::N,
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)
  ContinuousUnivariateParameterState{N}(
    value, v, v, v, v, v, v, v, v, v, v, v, v, diagnostickeys, diagnosticvalues
  )
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(
  ::Type{N},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousUnivariateParameterState(convert(N, NaN), diagnostickeys, diagnosticvalues)

Base.eltype{N<:FloatingPoint}(::Type{ContinuousUnivariateParameterState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousUnivariateParameterState{N}) = N

## ContinuousMultivariateParameterState

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ContinuousParameterState{Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorloglikelihood::Matrix{N}
  tensorlogprior::Matrix{N}
  tensorlogtarget::Matrix{N}
  dtensorloglikelihood::Array{N, 3}
  dtensorlogprior::Array{N, 3}
  dtensorlogtarget::Array{N, 3}
  size::Int
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Vector
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(
  value::Vector{N},
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)

  s = length(value)

  l = Array(Int, 9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Int) : s)
  end

  ContinuousMultivariateParameterState{N}(
    value,
    v,
    v,
    v,
    Array(N, l[1]),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4], l[4]),
    Array(N, l[5], l[5]),
    Array(N, l[6], l[6]),
    Array(N, l[7], l[7], l[7]),
    Array(N, l[8], l[8], l[8]),
    Array(N, l[9], l[9], l[9]),
    s,
    diagnostickeys,
    diagnosticvalues
  )
end

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousMultivariateParameterState(
    value, [main_cpstate_fields[i] in monitor ? true : false for i in 5:13], diagnostickeys, diagnosticvalues
  )

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  ::Type{N},
  size::Int=0,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

Base.eltype{N<:FloatingPoint}(::Type{ContinuousMultivariateParameterState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousMultivariateParameterState{N}) = N
