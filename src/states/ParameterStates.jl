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
  tensorloglikelihood::N
  tensorlogprior::N
  tensorlogtarget::N
  dtensorloglikelihood::N
  dtensorlogprior::N
  dtensorlogtarget::N
  diagnostics::Dict
end

function ContinuousUnivariateParameterState{N<:FloatingPoint}(value::N, diagnostics::Dict=Dict())
  v = convert(N, NaN)
  ContinuousUnivariateParameterState{N}(value, v, v, v, v, v, v, v, v, v, v, v, v, diagnostics)
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(::Type{N}, diagnostics::Dict=Dict()) =
  ContinuousUnivariateParameterState(convert(N, NaN), diagnostics)

Base.eltype{N<:FloatingPoint}(::Type{ContinuousUnivariateParameterState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousUnivariateParameterState{N}) = N

## ContinuousMultivariateParameterState

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ParameterState{Multivariate, N}
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
  diagnostics::Dict
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(
  value::Vector{N},
  monitor::Vector{Bool}=fill(false, 9),
  diagnostics::Dict=Dict()
)
  s = length(value)

  l = Array(Int, 9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Int) : s)
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
    Array(N, l[5], l[5]),
    Array(N, l[6], l[6]),
    Array(N, l[7], l[7], l[7]),
    Array(N, l[8], l[8], l[8]),
    Array(N, l[9], l[9], l[9]),
    s,
    diagnostics
  )
end

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostics::Dict=Dict()
) =
  ContinuousMultivariateParameterState(
    value, [main_state_field_names[i] in monitor ? true : false for i in 5:13], diagnostics
  )

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  ::Type{N},
  size::Int=0,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostics::Dict=Dict()
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostics)

ContinuousMultivariateParameterState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  monitor::Vector{Symbol},
  diagnostics::Dict=Dict()
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostics)

Base.eltype{N<:FloatingPoint}(::Type{ContinuousMultivariateParameterState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousMultivariateParameterState{N}) = N
