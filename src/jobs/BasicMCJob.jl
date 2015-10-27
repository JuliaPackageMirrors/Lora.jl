### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  runner::BasicMCRunner
  model::GenericModel # Model of a single parameter residing on the first node of model.vertices
  sampler::MCSampler
  tuner::MCTuner
  pindex::Int # Index of parameter in vstates
  iooptions::Array{Union{Dict, Void}}
  vstates::Vector{VariableState} # Vector of variable states ordered in accordance with variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  vnstates::Vector{Union{VariableNState, VariableIOStream, Void}}
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  send::Function
  receive::Function
  reset::Function
  save::Function
  close::function
  checkin::Bool # If checkin=true then check validity of job constructors' input arguments, else don't check

  function BasicMCJob(
    runner::BasicMCRunner,
    model::GenericModel,
    sampler::MCSampler,
    tuner::MCTuner,
    vstates::Vector{VariableState},
    pindex::Int,
    vnstates::Vector{Union{VariableNState, VariableIOStream, Void}},
    plain::Bool,
    checkin::Bool
  )
    instance = new()

    instance.runner = runner
    instance.model = model
    instance.sampler = sampler
    instance.tuner = tuner

    instance
  end
end

# Note: it is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
