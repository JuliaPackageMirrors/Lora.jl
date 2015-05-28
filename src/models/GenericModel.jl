typealias
  GenericModel
  GenericGraph{Variable, Dependence, Vector{Variable}, Vector{Dependence}, Vector{Vector{Dependence}}}

GenericModel(vs::Vector{Variable}, ds::Vector{Dependence}; is_directed::Bool=false) =
  graph(vs, ds, is_directed=is_directed)

GenericModel(is_directed::Bool=false) = graph(Variable[], Dependence[], is_directed=is_directed)

# Note 1: Create Dict{Symbol, KeyVertex{Symbol}} as an auxiliary dictionary to build the Model in the parsing stage
# Note 2: Later create Dict{KeyVertex{Symbol}, VariableState}(), to store value in VariableState.value
