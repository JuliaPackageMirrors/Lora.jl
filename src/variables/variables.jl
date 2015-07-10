### Sampleability

abstract Sampleability

type Deterministic <: Sampleability end

type Random <: Sampleability end

### Variable

abstract Variable{S<:Sampleability}

vertex_index(v::Variable) = v.index

convert(::Type{KeyVertex}, v::Variable) = KeyVertex{Symbol}(v.index, v.key)

convert(::Type{Vector{KeyVertex}}, v::Vector{Variable}) = KeyVertex{Symbol}[convert(KeyVertex, i) for i in v]

show(io::IO, v::Variable) = print(io, "Variable [$(v.index)]: $(v.key) ($(typeof(v)))")

### Deterministic Variable subtypes

## Constant

immutable Constant <: Variable{Deterministic}
  index::Int
  key::Symbol
end

## Hyperparameter

typealias Hyperparameter Constant

## Data

immutable Data <: Variable{Deterministic}
  index::Int
  key::Symbol
  update::Union(Function, Nothing)
end

Data(index::Int, key::Symbol) = Data(index, key, nothing)

## Transformation

immutable Transformation <: Variable{Deterministic}
  index::Int
  key::Symbol
  transform::Function
end
