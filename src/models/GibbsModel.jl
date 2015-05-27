type GibbsModel{Variable, Dependence} <: AbstractGraph{Variable, Dependence}
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

# Note: More importantly, define the inverse conversion, once the relevant constructors are in place
