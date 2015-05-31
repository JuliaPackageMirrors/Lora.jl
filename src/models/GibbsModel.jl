type GibbsModel <: AbstractGraph{Variable, Dependence}
  is_directed::Bool
  vertices::Vector{Variable}             # An indexable container of vertices (variables)
  edges::Vector{Dependence}              # An indexable container of edges (dependencies)
  finclist::Vector{Vector{Dependence}}   # Forward incidence list
  binclist::Vector{Vector{Dependence}}   # Nackward incidence list
  indexof::Dict{Variable, Int}           # Dictionary storing index for vertex (variable)
  stateof::Dict{Variable, VariableState} # Dictionary storing state for vertex (variable)
end

convert(::Type{GenericModel}, m::GibbsModel) =
  GenericModel(m.is_directed, m.vertices, m.edges, m.finclist, m.binclist, m.indexof)

function convert(::Type{GibbsModel}, m::GenericModel)
  s = Dict{Variable, VariableState}()
  for v in m.vertices
    s[v] = v.state
  end

  GibbsModel(m.is_directed, m.vertices, m.edges, m.finclist, m.binclist, m.indexof, s)
end

function add_vertex!(m::GibbsModel, v::Variable)
    push!(m.vertices, v)
    push!(m.finclist, Int[])
    push!(m.binclist, Int[])
    m.indexof[v] = length(m.vertices)
    v
end
