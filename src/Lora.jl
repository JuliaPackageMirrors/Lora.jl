module Lora

using Distributions
using Graphs

import Base:
  convert,
  show

import Graphs: 
  vertex_index,
  edge_index,
  source,
  target,
  revedge,
  is_directed,
  num_vertices,
  vertices,
  num_edges,
  edges,
  make_edge,
  out_edges,
  out_degree,
  out_neighbors,
  in_edges,
  in_degree,
  in_neighbors,
  add_vertex!,
  add_edge!

export
  ### Types
  Sampleability,
  Deterministic,
  Random,
  Variable,
  Constant,
  Hyperparameter,
  Data,
  Transformation,
  ContinuousUnivariateParameter,
  ContinuousMultivariateParameter,
  Dependence,
  VariableState,
  GenericVariableState,
  UnivariateGenericVariableState,
  MultivariateGenericVariableState,
  MatrixvariateGenericVariableState,
  ParameterState,
  ContinuousUnivariateParameterState,
  ContinuousMultivariateParameterState,
  GenericModel,
  ### Functions
  vertex_index,
  convert,
  show,
  edge_index,
  source,
  target,
  revedge,
  is_directed,
  num_vertices,
  vertices,
  num_edges,
  edges,
  make_edge,
  out_edges,
  out_degree,
  out_neighbors,
  in_edges,
  in_degree,
  in_neighbors,
  add_vertex!,
  add_edge!

include("variables/variables.jl")
include("variables/parameters.jl")
include("variables/dependencies.jl")
include("variables/states.jl")
include("models/GenericModel.jl")

end
