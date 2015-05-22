immutable UnivariateParameter{S<:ValueSupport} <: Parameter{Univariate, S}
  index::Int
  key::Symbol
  distribution::Union(UnivariateDistribution{S}, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
end

immutable MultivariateParameter{S<:ValueSupport} <: Parameter{Multivariate, S}
  index::Int
  key::Symbol
  distribution::Union(MultivariateDistribution{S}, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
end
