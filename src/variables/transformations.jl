abstract TransformationState{F<:VariateForm, N<:Number} <: VariableState{F, N, Deterministic}

type UnivariateTransformationState{N<:Number} <: TransformationState{Univariate, N}
  value::N
end

type MultivariateTransformationState{N<:Number} <: TransformationState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateTransformationState{N<:Number}(value::Vector{N}) = MultivariateTransformationState{N}(value, length(value))

MultivariateTransformationState{N<:Number}(::Type{N}, size::Int=0) =
  MultivariateTransformationState{N}(Array(N, size), size)

type MatrixvariateTransformationState{N<:Number} <: TransformationState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple
end

MatrixvariateTransformationState{N<:Number}(value::Matrix{N}) = MatrixvariateTransformationState{N}(value, size(value))

MatrixvariateTransformationState{N<:Number}(::Type{N}, size::Int=0) =
  MatrixvariateTransformationState{N}(Array(N, size, size), size)

abstract Transformation{F<:VariateForm, N<:Number} <: Variable{F, N, Deterministic}

type UnivariateTransformation{N<:Number} <: Transformation{Univariate, N}
  index::Int
  key::Symbol
  transform::Function
  state::UnivariateTransformationState{N}
end

type MultivariateTransformation{N<:Number} <: Transformation{Multivariate, N}
  index::Int
  key::Symbol
  transform::Function
  state::MultivariateTransformationState{N}
end

MultivariateTransformation{N<:Number}(index::Int, key::Symbol, transform::Function, value::Vector{N}) =
  MultivariateTransformation{N}(index, key, transform, MultivariateTransformationState(value))

MultivariateTransformation{N<:Number}(index::Int, key::Symbol, transform::Function, ::Type{N}, size::Int=0) =
  MultivariateTransformation{N}(index, key, transform, MultivariateTransformationState(size))

type MatrixvariateTransformation{N<:Number} <: Transformation{Multivariate, N}
  index::Int
  key::Symbol
  transform::Function
  state::MatrixvariateTransformationState{N}
end

MatrixvariateTransformation{N<:Number}(index::Int, key::Symbol, transform::Function, value::Matrix{N}) =
  MatrixvariateTransformation{N}(index, key, transform, MatrixvariateTransformationState(value))

MatrixvariateTransformation{N<:Number}(index::Int, key::Symbol, transform::Function, ::Type{N}, size::Int=0) =
  MatrixvariateTransformation{N}(index, key, transform, MatrixvariateTransformationState(size))
