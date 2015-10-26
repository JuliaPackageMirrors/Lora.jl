module Lora

using Distributions
using Graphs

import Base:
  ==,
  close,
  convert,
  copy!,
  eltype,
  isequal,
  read,
  read!,
  run,
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
  MCRunner,
  VariableState,
  BasicVariableState,
  UnivariateBasicVariableState,
  MultivariateBasicVariableState,
  MatrixvariateBasicVariableState,
  ParameterState,
  ContinuousParameterState,
  ContinuousUnivariateParameterState,
  ContinuousMultivariateParameterState,
  VariableNState,
  BasicVariableNState,
  UnivariateBasicVariableNState,
  MultivariateBasicVariableNState,
  MatrixvariateBasicVariableNState,
  ParameterNState,
  ContinuousParameterNState,
  MCChain,
  ContinuousMCChain,
  ContinuousUnivariateParameterNState,
  ContinuousUnivariateMCChain,
  ContinuousMultivariateParameterNState,
  ContinuousMultivariateMCChain,
  VariableIOStream,
  BasicVariableIOStream,
  ParameterIOStream,
  ContinuousParameterIOStream,
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
  MCTune,
  BasicMCTune,
  MCTuner,
  VanillaMCTuner,
  AcceptanceRateMCTuner,
  MCSamplerState,
  MCSampler,
  MHSampler,
  HMCSampler,
  LMCSampler,
  MHState,
  MH,
  BasicMCRunner,
  MCJob,
  ### Functions
  logistic,
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
  single_parameter_likelihood_model,
  reset!,
  count!,
  rate!,
  logistic_rate_score,
  erf_rate_score,
  tune!,
  run

include("common.jl")

include("states/VariableStates.jl")
include("states/ParameterStates.jl")
include("states/VariableNStates.jl")
include("states/ParameterNStates.jl")
include("iostreams/VariableIOStreams.jl")
include("iostreams/ParameterIOStreams.jl")
include("variables/variables.jl")
include("variables/parameters.jl")
include("variables/dependencies.jl")
include("models/GenericModel.jl")
include("models/generators.jl")

include("tuners/tuners.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/AcceptanceRateMCTuner.jl")
include("samplers/samplers.jl")
include("samplers/MH.jl")

include("runners/BasicMCRunner.jl")

include("jobs/jobs.jl")
# include("jobs/BasicMCJob.jl")
# include("jobs/GibbsJob.jl")

end
