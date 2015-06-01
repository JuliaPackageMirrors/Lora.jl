immutable Dependence{S<:Variable, T<:Variable}
  index::Int
  source::S
  target::T
end

edge_index(d::Dependence) = d.index
source(d::Dependence) = e.source
target(d::Dependence) = e.target

revedge{S<:Variable, T<:Variable}(d::Dependence{S, T}) = Dependence(d.index, d.target, d.source)
