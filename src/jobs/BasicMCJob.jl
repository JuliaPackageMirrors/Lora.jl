### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  runner::BasicMCRunner
  model::GenericModel # Model of a single parameter residing on the first node of model.vertices
  sampler::MCSampler
  tuner::MCTuner
  vstates::Vector{VariableState} # Vector of variable states ordered in accordance with variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  inmemory::Bool # If inmemory=true then store output in memory via NStates, else in files via IOStreams
  checkin::Bool # If checkin=true then check validity of job constructors' input arguments, else don't check
  task::Union{Task, Void}
  send::Function
  receive::Function
  reset::Function
  save::Function
  close::function

  # function BasicMCJob(m::GenericModel s::MCSampler r::MCRunner t::MCTuner)
  #   If checkin=true then check isa(model, likelihood) and isa(model.vertices[1], Parameter)
  # end
end

# Note: it is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
