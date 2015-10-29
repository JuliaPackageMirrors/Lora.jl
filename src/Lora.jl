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
  reset,
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
  AcceptanceRateMCTuner,
  BasicMCJob,
  BasicMCRunner,
  BasicMCTune,
  BasicVariableIOStream,
  BasicVariableNState,
  BasicVariableState,
  Constant,
  ContinuousMCChain,
  ContinuousMultivariateMCChain,
  ContinuousMultivariateParameter,
  ContinuousMultivariateParameterNState,
  ContinuousMultivariateParameterState,
  ContinuousParameter,
  ContinuousParameterIOStream,
  ContinuousParameterNState,
  ContinuousParameterState,
  ContinuousUnivariateMCChain,
  ContinuousUnivariateParameter,
  ContinuousUnivariateParameterNState,
  ContinuousUnivariateParameterState,
  Data,
  Dependence,
  Deterministic,
  GenericModel,
  HMCSampler,
  Hyperparameter,
  LMCSampler,
  MCChain,
  MCJob,
  MCRunner,
  MCSampler,
  MCSamplerState,
  MCTune,
  MCTuner,
  MH,
  MHSampler,
  MHState,
  MatrixvariateBasicVariableNState,
  MatrixvariateBasicVariableState,
  MultivariateBasicVariableNState,
  MultivariateBasicVariableState,
  Parameter,
  ParameterIOStream,
  ParameterNState,
  ParameterState,
  Random,
  Sampleability,
  Transformation,
  UnivariateBasicVariableNState,
  UnivariateBasicVariableState,
  VanillaMCTuner,
  Variable,
  VariableIOStream,
  VariableNState,
  VariableState,

  ### Functions
  add_dimension,
  add_edge!,
  add_vertex!,
  augment!,
  count!,
  edge_index,
  edges,
  erf_rate_score,
  in_degree,
  in_edges,
  in_neighbors,
  is_directed,
  likelihood_model,
  logistic,
  logistic_rate_score,
  make_edge,
  minify,
  num_edges,
  num_vertices,
  out_degree,
  out_edges,
  out_neighbors,
  rate!,
  reset!,
  revedge,
  run,
  sampler_state,
  save!,
  save,
  single_parameter_likelihood_model,
  source,
  target,
  topological_sort_by_dfs,
  tune!,
  tune_state,
  variable_nstate,
  vertex_index,
  vertices

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
include("jobs/BasicMCJob.jl")
# include("jobs/GibbsJob.jl")

end
