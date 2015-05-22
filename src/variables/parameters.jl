# It may be decided to add a domain field in Parameter sub-types

immutable UnivariateParameter{S<:ValueSupport} <: Parameter{Univariate, S}
  index::Int
  key::Symbol
  size::Int
  distribution::Union(UnivariateDistribution{S}, Nothing)
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
end

# Do not forget to force input argument types for function fields, as in x::Float64->f(x)  

immutable MultivariateParameter{S<:ValueSupport} <: Parameter{Multivariate, S}
  index::Int
  key::Symbol
  size::Int
  distribution::Union(MultivariateDistribution{S}, Nothing)
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
end
