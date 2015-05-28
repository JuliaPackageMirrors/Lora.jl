abstract DataState{F<:VariateForm, N<:Number} <: VariableState{F, N, Deterministic}

type UnivariateDataState{N<:Number} <: DataState{Univariate, N}
  value::N
end

type MultivariateDataState{N<:Number} <: DataState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateDataState{N<:Number}(value::Vector{N}) = MultivariateDataState{N}(value, length(value))

MultivariateDataState{N<:Number}(::Type{N}, size::Int=0) = MultivariateDataState{N}(Array(N, size), size)

type MatrixvariateDataState{N<:Number} <: DataState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple
end

MatrixvariateDataState{N<:Number}(value::Matrix{N}) = MatrixvariateDataState{N}(value, size(value))

MatrixvariateDataState{N<:Number}(::Type{N}, size::Int=0) = MatrixvariateDataState{N}(Array(N, size, size), size)

abstract Data{F<:VariateForm, N<:Number} <: Variable{F, N, Deterministic}

type UnivariateData{N<:Number} <: Data{Univariate, N}
  index::Int
  key::Symbol
  state::UnivariateDataState{N}
end

type MultivariateData{N<:Number} <: Data{Multivariate, N}
  index::Int
  key::Symbol
  state::MultivariateDataState{N}
end

MultivariateData{N<:Number}(index::Int, key::Symbol, value::Vector{N}) =
  MultivariateData{N}(index, key, MultivariateDataState(value))

MultivariateData{N<:Number}(index::Int, key::Symbol, ::Type{N}, size::Int=0) =
  MultivariateData{N}(index, key, MultivariateDataState(size))

type MatrixvariateData{N<:Number} <: Data{Multivariate, N}
  index::Int
  key::Symbol
  state::MatrixvariateDataState{N}
end

MatrixvariateData{N<:Number}(index::Int, key::Symbol, value::Matrix{N}) =
  MatrixvariateData{N}(index, key, MatrixvariateDataState(value))

MatrixvariateData{N<:Number}(index::Int, key::Symbol, ::Type{N}, size::Int=0) =
  MatrixvariateData{N}(index, key, MatrixvariateDataState(size))
