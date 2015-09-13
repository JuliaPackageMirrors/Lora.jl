module Lora

using Distributions
using Graphs

import Base:
  close,
  convert,
  copy!,
  eltype,
  read,
  read!,
  show,
  write

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
  add_edge!,
  topological_sort_by_dfs

export
  ### Types
  VariableState,
  GenericVariableState,
  UnivariateGenericVariableState,
  MultivariateGenericVariableState,
  MatrixvariateGenericVariableState,
  ParameterState,
  ContinuousUnivariateParameterState,
  ContinuousMultivariateParameterState,
  VariableNState,
  GenericVariableNState,
  ParameterNState,
  MCChain,
  UnivariateGenericVariableNState,
  MultivariateGenericVariableNState,
  MatrixvariateGenericVariableNState,
  ContinuousUnivariateParameterNState,
  ContinuousUnivariateMCChain,
  ContinuousMultivariateParameterNState,
  ContinuousMultivariateMCChain,
  VariableIOStream,
  GenericVariableIOStream,
  Sampleability,
  Deterministic,
  Random,
  Variable,
  Constant,
  Hyperparameter,
  Data,
  Transformation,
  Parameter,
  ContinuousUnivariateParameter,
  ContinuousMultivariateParameter,
  Dependence,
  GenericModel,
  ### Functions
  add_dimension,
  save!,
  save,
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
  add_edge!,
  topological_sort_by_dfs,
  likelihood_model,
  single_parameter_likelihood_model

include("states/VariableStates.jl")
include("states/VariableNStates.jl")
include("states/ParameterNStates.jl")
include("iostreams/VariableIOStreams.jl")
# include("iostreams/ParameterIOStreams.jl")
include("variables/variables.jl")
include("variables/parameters.jl")
include("variables/dependencies.jl")
include("models/GenericModel.jl")
include("models/generators.jl")

end
