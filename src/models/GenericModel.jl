typealias
  GenericModel
  GenericGraph{Variable, Dependence, Vector{Variable}, Vector{Dependence}, Vector{Vector{Dependence}}}

GenericModel(vs::Vector{Variable}, ds::Vector{Dependence}; is_directed::Bool=false) =
  graph(vs, ds, is_directed=is_directed)

GenericModel(is_directed::Bool=false) = graph(Variable[], Dependence[], is_directed=is_directed)

function add_vertex!(m::GenericModel, v::Variable)
    push!(m.vertices, v)
    push!(m.finclist, Int[])
    push!(m.binclist, Int[])
    m.indexof[v] = length(m.vertices)
    v
end
