### Generators are functions used for generating various specific models as instances of GenericModel

## likelihood_model represents a likelihood L(Vector{Parameter} | Vector{Data}, Vector{Hyperparameter})

function likelihood_model{P<:Parameter}(
  p::Vector{P};
  data::Vector{Data}=Array(Data, 0),
  hyperparameters::Vector{Hyperparameter}=Array(Hyperparameter, 0),
  is_directed::Bool=true
)
  m = GenericModel(is_directed)

  for v in [p; data; hyperparameters]
    add_vertex!(m, v)
    m.indexof[v] = v.index
  end

  for t in p
    for s in [data; hyperparameters]
      add_edge!(m, s, t)
    end
  end

  return m
end

## single_parameter_likelihood_model represents a likelihood L(Parameter | Vector{Data}, Vector{Hyperparameter})

single_parameter_likelihood_model(
  p::Parameter;
  data::Vector{Data}=Array(Data, 0),
  hyperparameters::Vector{Hyperparameter}=Array(Hyperparameter, 0),
  is_directed::Bool=false
) =
  likelihood_model(Parameter[p], data=data, hyperparameters=hyperparameters, is_directed=is_directed)
