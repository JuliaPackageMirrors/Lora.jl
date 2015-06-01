type GenericModel <: AbstractGraph{Variable, Dependence}
  is_directed::Bool
  vertices::Vector{Variable}             # An indexable container of vertices (variables)
  edges::Vector{Dependence}              # An indexable container of edges (dependencies)
  finclist::Vector{Vector{Dependence}}   # Forward incidence list
  binclist::Vector{Vector{Dependence}}   # Nackward incidence list
  indexof::Dict{Variable, Int}           # Dictionary storing index for vertex (variable)
  stateof::Dict{Variable, VariableState} # Dictionary storing state for vertex (variable)
end

is_directed(m::GenericModel) = m.is_directed

num_vertices(m::GenericModel) = length(m.vertices)
vertices(m::GenericModel) = m.vertices

num_edges(m::GenericModel) = length(m.edges)
edges(m::GenericModel) = m.edges

vertex_index(v::Integer, m::GenericModel) = (v <= m.vertices[end] ? v : 0)
vertex_index(v::Variable, m::GenericModel) = vertex_index(v)

edge_index(d::Dependence, m::GenericModel) = edge_index(d)

make_edge(m::GenericModel, s::Variable, t::Variable) = Dependence(num_edges(m)+1, s, t)

revedge{S<:Variable, T<:Variable}(d::Dependence{S, T}) = Dependence(d.index, d.target, d.source)

out_edges(v::Variable, m::GenericModel) = m.finclist[vertex_index(v, m)]
out_degree(v::Variable, m::GenericModel) = length(out_edges(v, m))
out_neighbors(v::Variable, m::GenericModel) = TargetIterator(m, out_edges(v, m))

in_edges(v::Variable, m::GenericModel) = m.binclist[vertex_index(v, m)]
in_degree(v::Variable, m::GenericModel) = length(in_edges(v, m))
in_neighbors(v::Variable, m::GenericModel) = SourceIterator(m, in_edges(v, m))

function add_vertex!(m::GenericModel, v::Variable)
    push!(m.vertices, v)
    push!(m.finclist, Int[])
    push!(m.binclist, Int[])
    m.indexof[v] = length(m.vertices)
    m.stateof[v] = v.state
    v
end

function add_edge!(m::GenericModel, u::Variable, v::Variable, d::Dependence)
    ui = vertex_index(u, m)::Int
    vi = vertex_index(v, m)::Int

    push!(m.edges, d)
    push!(m.finclist[ui], d)
    push!(m.binclist[vi], d)

    if !m.is_directed
        rev_d = revedge(d)
        push!(m.finclist[vi], rev_d)
        push!(m.binclist[ui], rev_d)
    end
    
    d
end

add_edge!(m::GenericModel, d::Dependence) = add_edge!(m, source(d, m), target(d, m), d)
add_edge!(m::GenericModel, u::Variable, v::Variable) = add_edge!(m, u, v, make_edge(m, u, v))

function GenericModel(vs::Vector{Variable}, ds::Vector{Dependence}; is_directed::Bool=false)
  n = length(vs)
  m = GenericModel(
    is_directed,
    Variable[],
    Dependence[],
    Graphs.multivecs(Dependence, n),
    Graphs.multivecs(Dependence, n),
    Dict{Variable, Int}(),
    Dict{Variable, VariableState}()
  )

  for v in vs
    add_vertex!(m, v)
    m.indexof[v] = v.index
    m.stateof[v] = v.state
  end
  
  for d in ds
    add_edge!(m, d)
  end
  
  return m
end

GenericModel(is_directed::Bool=false) = GenericModel(Variable[], Dependence[], is_directed=is_directed)
