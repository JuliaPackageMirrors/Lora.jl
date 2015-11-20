module Lora

using Distributions
using Graphs

import Base:
  ==,
  close,
  convert,
  copy!,
  eltype,
  flush,
  getindex,
  isequal,
  mark,
  open,
  read!,
  read,
  run,
  show,
  write

import Graphs:
  add_edge!,
  add_vertex!,
  edge_index,
  edges,
  in_degree,
  in_edges,
  in_neighbors,
  is_directed,
  make_edge,
  num_edges,
  num_vertices,
  out_degree,
  out_edges,
  out_neighbors,
  revedge,
  source,
  target,
  topological_sort_by_dfs,
  vertex_index,
  vertices

export
  ### Types
  AcceptanceRateMCTuner,
  BasicMCJob,
  BasicMCRange,
  BasicMCTune,
  BasicVariableIOStream,
  BasicVariableNState,
  BasicVariableState,
  Constant,
  ContinuousMarkovChain,
  ContinuousMultivariateMarkovChain,
  ContinuousMultivariateParameter,
  ContinuousMultivariateParameterNState,
  ContinuousMultivariateParameterState,
  ContinuousParameter,
  ContinuousParameterIOStream,
  ContinuousParameterNState,
  ContinuousParameterState,
  ContinuousUnivariateMarkovChain,
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
  MarkovChain,
  MCJob,
  MCRange,
  MCSampler,
  MCSamplerState,
  MCTuner,
  MCTunerState,
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
  count!,
  edge_index,
  edges,
  erf_rate_score,
  in_degree,
  in_edges,
  in_neighbors,
  is_directed,
  is_indexed,
  likelihood_model,
  logistic,
  logistic_rate_score,
  make_edge,
  num_edges,
  num_vertices,
  out_degree,
  out_edges,
  out_neighbors,
  rate!,
  replace!,
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
  tuner_state,
  vertex_index,
  vertices

include("parser/replace.jl")

include("stats/logistic.jl")

include("states/VariableStates.jl")
include("states/ParameterStates.jl")
include("states/VariableNStates.jl")
include("states/ParameterNStates.jl")

include("iostreams/VariableIOStreams.jl")
include("iostreams/ParameterIOStreams.jl")

include("variables/variables.jl")
include("variables/ContinuousUnivariateParameter.jl")
include("variables/ContinuousMultivariateParameter.jl")
include("variables/dependencies.jl")

include("models/GenericModel.jl")
include("models/generators.jl")

include("ranges/ranges.jl")
include("ranges/BasicMCRange.jl")

include("tuners/tuners.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/AcceptanceRateMCTuner.jl")

include("samplers/samplers.jl")
include("samplers/MH.jl")

include("jobs/jobs.jl")
include("jobs/BasicMCJob.jl")
# include("jobs/GibbsJob.jl")

include("samplers/iterate/MH.jl")
include("samplers/iterate/iterate.jl")

end
