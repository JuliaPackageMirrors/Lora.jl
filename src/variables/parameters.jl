typealias Parameter{F<:VariateForm, S<:ValueSupport} Variable{F, S, Random}

immutable UnivariateParameter{S<:ValueSupport} <: Parameter{Univariate, S}
  key::Symbol
  distribution::Union(UnivariateDistribution{S}, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
end

type MultivariateParameter{S<:ValueSupport} <: Parameter{Multivariate, S}
  key::Symbol
  distribution::Union(MultivariateDistribution{S}, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
end
