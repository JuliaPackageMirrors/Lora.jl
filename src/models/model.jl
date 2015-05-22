typealias Model GenericGraph{Variable, Dependence, Vector{Variable}, Vector{Dependence}, Vector{Vector{Dependence}}}

Model(vs::Vector{Variable}, ds::Vector{Dependence}; is_directed::Bool=true) = graph(vs, ds, is_directed=is_directed)

Model(is_directed::Bool=true) = graph(Variable[], Dependence[], is_directed=is_directed)
