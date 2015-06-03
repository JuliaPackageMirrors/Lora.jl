abstract ConstantState{F<:VariateForm, N<:Number} <: VariableState{F, N, Deterministic}

immutable UnivariateConstantState{N<:Number} <: ConstantState{Univariate, N}
  value::N
end

immutable MultivariateConstantState{N<:Number} <: ConstantState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateConstantState{N<:Number}(value::Vector{N}) = MultivariateConstantState{N}(value, length(value))

abstract Constant{F<:VariateForm, N<:Number} <: Variable{F, N, Deterministic}

type UnivariateConstant{N<:Number} <: Constant{Univariate, N}
  index::Int
  key::Symbol
  state::UnivariateConstantState{N}
end

type MultivariateConstant{N<:Number} <: Constant{Multivariate, N}
  index::Int
  key::Symbol
  state::MultivariateConstantState{N}
end

typealias HyperparameterState{F<:VariateForm, N<:Number} ConstantState{F, N}

typealias UnivariateHyperparameterState{N<:Number} UnivariateConstantState{N}

typealias MultivariateHyperparameterState{N<:Number} MultivariateConstantState{N}

typealias Hyperparameter{F<:VariateForm, N<:Number} Constant{F, N}

typealias UnivariateHyperparameter{N<:Number} UnivariateConstant{N}

typealias MultivariateHyperparameter{N<:Number} MultivariateConstant{N}
