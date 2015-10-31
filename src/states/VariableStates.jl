### Abstract variable states

abstract VariableState{F<:VariateForm, N<:Number}

abstract BasicVariableState{F<:VariateForm, N<:Number} <: VariableState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{BasicVariableState{F, N}}) = N

### Basic variable state subtypes

## UnivariateBasicVariableState

type UnivariateBasicVariableState{N<:Number} <: BasicVariableState{Univariate, N}
  value::N
end

Base.eltype{N<:Number}(::Type{UnivariateBasicVariableState{N}}) = N
Base.eltype{N<:Number}(s::UnivariateBasicVariableState{N}) = N

## MultivariateBasicVariableState

type MultivariateBasicVariableState{N<:Number} <: BasicVariableState{Multivariate, N}
  value::Vector{N}
  size::Int
end

MultivariateBasicVariableState{N<:Number}(value::Vector{N}) = MultivariateBasicVariableState{N}(value, length(value))

MultivariateBasicVariableState{N<:Number}(size::Int, ::Type{N}=Float64) =
  MultivariateBasicVariableState{N}(Array(N, size), size)

Base.eltype{N<:Number}(::Type{MultivariateBasicVariableState{N}}) = N
Base.eltype{N<:Number}(s::MultivariateBasicVariableState{N}) = N

## MatrixvariateBasicVariableState

type MatrixvariateBasicVariableState{N<:Number} <: BasicVariableState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple{Int, Int}
end

MatrixvariateBasicVariableState{N<:Number}(value::Matrix{N}) = MatrixvariateBasicVariableState{N}(value, size(value))

MatrixvariateBasicVariableState{N<:Number}(size::Tuple, ::Type{N}=Float64) =
  MatrixvariateBasicVariableState{N}(Array(N, size...), size)

Base.eltype{N<:Number}(::Type{MatrixvariateBasicVariableState{N}}) = N
Base.eltype{N<:Number}(s::MatrixvariateBasicVariableState{N}) = N
