abstract ParameterState{S<:ValueSupport, F<:VariateForm, N<:Number} <: VariableState{F, N, Random}

type ContinuousUnivariateParameterState{N<:FloatingPoint} <: ParameterState{Continuous, Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradlogtarget::N
  tensorlogtarget::N
  dtensorlogtarget::N
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(value::N) =
  ContinuousUnivariateParameterState{N}(
    value,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN)
  )

ContinuousUnivariateParameterState() = ContinuousUnivariateParameterState(NaN, NaN, NaN, NaN, NaN, NaN, NaN)

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ParameterState{Continuous, Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradlogtarget::Vector{N}
  tensorlogtarget::Matrix{N}
  dtensorlogtarget::Array{N, 3}
  size::Int
end

ContinuousMultivariateParameterState{N<:FloatingPoint}(value::Vector{N}) =
  ContinuousMultivariateParameterState{N}(
    value,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    Array(N, 0),
    Array(N, 0, 0),
    Array(N, 0, 0, 0),
    length(value)
  )

ContinuousMultivariateParameterState() =
  ContinuousMultivariateParameterState(
    Array(Float64, 0),
    NaN,
    NaN,
    NaN,
    Array(Float64, 0),
    Array(Float64, 0, 0),
    Array(Float64, 0, 0, 0),
    0
  )

ContinuousMultivariateParameterState(size::Int) =
  ContinuousMultivariateParameterState(
    Array(Float64, size),
    NaN,
    NaN,
    NaN,
    Array(Float64, size),
    Array(Float64, size, size),
    Array(Float64, size, size, size),
    size
  )

typealias Parameter{S<:ValueSupport, F<:VariateForm, N<:Number} Variable{F, N, Random}

type ContinuousUnivariateParameter{N<:FloatingPoint} <: Parameter{Continuous, Univariate, N}
  index::Int
  key::Symbol
  distribution::Union(ContinuousUnivariateDistribution, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
  gradloglikelihood::Union(Function, Nothing)
  gradlogprior::Union(Function, Nothing)
  gradlogtarget::Union(Function, Nothing)
  tensorloglikelihood::Union(Function, Nothing)
  tensorlogprior::Union(Function, Nothing)
  tensorlogtarget::Union(Function, Nothing)
  dtensorloglikelihood::Union(Function, Nothing)
  dtensorlogprior::Union(Function, Nothing)
  dtensorlogtarget::Union(Function, Nothing)
  uptogradlogtarget::Union(Function, Nothing)
  uptotensorlogtarget::Union(Function, Nothing)
  uptodtensorlogtarget::Union(Function, Nothing)
  state::ContinuousUnivariateParameterState{N}
end

type ContinuousMultivariateParameter{N<:FloatingPoint} <: Parameter{Continuous, Multivariate, N}
  index::Int
  key::Symbol
  distribution::Union(ContinuousMultivariateDistribution, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
  gradloglikelihood::Union(Function, Nothing)
  gradlogprior::Union(Function, Nothing)
  gradlogtarget::Union(Function, Nothing)
  tensorloglikelihood::Union(Function, Nothing)
  tensorlogprior::Union(Function, Nothing)
  tensorlogtarget::Union(Function, Nothing)
  dtensorloglikelihood::Union(Function, Nothing)
  dtensorlogprior::Union(Function, Nothing)
  dtensorlogtarget::Union(Function, Nothing)
  uptogradlogtarget::Union(Function, Nothing)
  uptotensorlogtarget::Union(Function, Nothing)
  uptodtensorlogtarget::Union(Function, Nothing)
  state::ContinuousMultivariateParameterState{N}
end
