### Sampleability

abstract Sampleability

type Deterministic <: Sampleability end

type Random <: Sampleability end

### Variable

abstract Variable{S<:Sampleability}

Base.eltype{S<:Sampleability}(::Type{Variable{S}}) = S

vertex_index(v::Variable) = v.index

is_indexed(v::Variable) = v.index > 0 ? true : false

Base.convert(::Type{KeyVertex}, v::Variable) = KeyVertex{Symbol}(v.index, v.key)
Base.convert(::Type{Vector{KeyVertex}}, v::Vector{Variable}) = KeyVertex{Symbol}[convert(KeyVertex, i) for i in v]

Base.show(io::IO, v::Variable) = print(io, "Variable [$(v.index)]: $(v.key) ($(typeof(v)))")

### Deterministic Variable subtypes

## Constant

immutable Constant <: Variable{Deterministic}
  key::Symbol
  index::Int
end

Constant(key::Symbol) = Constant(key, 0)

default_state{N<:Number}(variable::Constant, value::N) = UnivariateBasicVariableState(value)
default_state{N<:Number}(variable::Constant, value::Vector{N}) = MultivariateBasicVariableState(value)
default_state{N<:Number}(variable::Constant, value::Matrix{N}) = MatrixvariateBasicVariableState(value)

## Hyperparameter

typealias Hyperparameter Constant

## Data

immutable Data <: Variable{Deterministic}
  key::Symbol
  index::Int
  update::Union{Function, Void}
end

Data(key::Symbol, index::Int) = Data(key, index, nothing)
Data(key::Symbol, update::Union{Function, Void}) = Data(key, 0, update)
Data(key::Symbol) = Data(key, 0, nothing)

default_state{N<:Number}(variable::Data, value::N) = UnivariateBasicVariableState(value)
default_state{N<:Number}(variable::Data, value::Vector{N}) = MultivariateBasicVariableState(value)
default_state{N<:Number}(variable::Data, value::Matrix{N}) = MatrixvariateBasicVariableState(value)

## Transformation

immutable Transformation <: Variable{Deterministic}
  key::Symbol
  index::Int
  transform::Function
end

Transformation(key::Symbol, transform::Function) = Transformation(key, 0, transform)

default_state{N<:Number}(variable::Transformation, value::N) = UnivariateBasicVariableState(value)
default_state{N<:Number}(variable::Transformation, value::Vector{N}) = MultivariateBasicVariableState(value)
default_state{N<:Number}(variable::Transformation, value::Matrix{N}) = MatrixvariateBasicVariableState(value)

### Random Variable subtypes

## Astract Parameter types

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

abstract ContinuousParameter{S<:ValueSupport, F<:VariateForm} <: Parameter{S, F}
