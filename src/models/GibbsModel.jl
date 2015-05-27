type GibbsModel{V, E, VList, EList, IncList} <: AbstractGraph{V, E}
  is_directed::Bool
  vertices::VList                        # An indexable container of vertices (variables)
  edges::EList                           # An indexable container of edges (dependencies)
  finclist::IncList                      # Forward incidence list
  binclist::IncList                      # Nackward incidence list
  indexof::Dict{V, Int}                  # Dictionary storing index for vertex (variable)
  stateof::Dict{Variable, VariableState} # Dictionary storing state for vertex (variable)
end

convert(::Type{GenericModel}, m::GibbsModel) =
  GenericModel(m.is_directed, m.vertices, m.edges, m.finclist, m.binclist, m.indexof)

# Note: More importantly, define the inverse conversion, once the relevant constructors are in place
