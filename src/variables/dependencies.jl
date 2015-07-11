### Dependence

immutable Dependence{S<:Variable, T<:Variable}
  index::Int
  source::S
  target::T
end

edge_index(d::Dependence) = d.index
source(d::Dependence) = e.source
target(d::Dependence) = e.target

revedge{S<:Variable, T<:Variable}(d::Dependence{S, T}) = Dependence(d.index, d.target, d.source)

convert(::Type{Edge}, d::Dependence) =
  Edge{KeyVertex{Symbol}}(d.index, convert(KeyVertex, d.source), convert(KeyVertex, d.target))

convert(::Type{Vector{Edge}}, d::Vector{Dependence}) = Edge{KeyVertex{Symbol}}[convert(Edge, i) for i in d]

show(io::IO, d::Dependence) = print(io, "Dependence [$(d.index)]: $(d.target.key) | $(d.source.key)")
