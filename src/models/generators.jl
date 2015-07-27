### Generators are functions used for generating various specific models as instances of GenericModel

##

function single_parameter_posterior(
  p::Parameter,
  data::Vector{Data},
  hp::Vector{Hyperparameter};
  is_directed::Bool=true
)
  m = GenericModel(is_directed)

  for v in [p, data, hp]
    add_vertex!(m, v)
    m.indexof[v] = v.index
  end

  for v in [data, hp]
    add_edge!(m, v, p)
  end

  return m
end
