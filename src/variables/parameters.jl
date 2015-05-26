abstract ParameterState{S<:ValueSupport, F<:VariateForm, N<:Number} <: VariableState{F, N, Random}

type ContinuousUnivariateParameterState{N<:Number} <: ParameterState{Continuous, Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradlogtarget::N
  tensorlogtarget::N
  dtensorlogtarget::N
end

type ContinuousMultivariateParameterState{N<:Number} <: ParameterState{Continuous, Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradlogtarget::Vector{N}
  tensorlogtarget::Matrix{N}
  dtensorlogtarget::Array{N, 3}
  size::Int
end

typealias Parameter{S<:ValueSupport, F<:VariateForm, N<:Number} Variable{F, N, Random}

immutable ContinuousUnivariateParameter{N<:Number} <: Parameter{Continuous, Univariate, N}
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

immutable ContinuousMultivariateParameter{N<:Number} <: Parameter{Continuous, Multivariate, N}
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
