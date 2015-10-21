### BasicMCJob is used for sampling a single parameter via serial Monte Carlo
### It is the most elementary and typical Markov chain Monte Carlo (MCMC) method

type BasicMCJob <: MCJob
  model::GenericModel # Model of a single parameter residing on the first node of model.vertices
  sampler::MCSampler
  runner::SerialMC
  tuner::MCTuner
  pstate::ContinuousParameterState # State of single parameter of model, i.e. of first node in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  send::Function
  receive::Function
  reset::Function
  plain::Bool # If job flow is controlled via tasks, then plain=false, otherwise plain=true
  task::Union{Task, Void}

  function BasicMCJob(m::GenericModel s::MCSampler r::MCRunner t::MCTuner)
  end
end
