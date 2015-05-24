typealias Model GenericGraph{Variable, Dependence, Vector{Variable}, Vector{Dependence}, Vector{Vector{Dependence}}}

Model(vs::Vector{Variable}, ds::Vector{Dependence}; is_directed::Bool=true) = graph(vs, ds, is_directed=is_directed)

Model(is_directed::Bool=true) = graph(Variable[], Dependence[], is_directed=is_directed)

# Create Dict{Symbol, KeyVertex{Symbol}} as an auxiliary dictionary to build the Model in the parsing stage
# Later create Dict{KeyVertex{Symbol}, VariableState}(), to store value in VariableState.value
