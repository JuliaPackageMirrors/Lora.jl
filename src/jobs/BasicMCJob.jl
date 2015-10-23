### BasicMCJob is used for sampling a single parameter via serial Monte Carlo
### It is the most elementary and typical Markov chain Monte Carlo (MCMC) strategy

type BasicMCJob <: MCJob
  model::GenericModel # Model of a single parameter residing on the first node of model.vertices
  runner::SerialMC
  sampler::MCSampler
  tuner::MCTuner
  vstates::Vector{VariableState} # Vector of variable states ordered in accordance with variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  send::Function
  receive::Function
  reset::Function
  plain::Bool # If job flow is controlled via tasks then plain=false, else plain=true
  task::Union{Task, Void}
  validate::Bool # To check validity of job constructors' input arguments set validate=true, else set validate=false

  # function BasicMCJob(m::GenericModel s::MCSampler r::MCRunner t::MCTuner)
  # If check=true, then check isa(model, likelihood) and isa(model.vertices[1], Parameter)
  # end
end
