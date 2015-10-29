### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  runner::BasicMCRunner
  model::GenericModel # Model of a single parameter
  sampler::MCSampler
  tuner::MCTuner
  pindex::Int # Index of single parameter in vstates
  vstates::Vector{VariableState} # Vector of variable states ordered according to variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  output::Union{VariableNState, VariableIOStream} # Output of model's single parameter
  # plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  # task::Union{Task, Void}
  # send::Function
  # receive::Function
  # reset::Function
  # save::Function
  # close::function
  # checkin::Bool # If checkin=true then check validity of job constructors' input arguments, else don't check

  BasicMCJob(
    runner::BasicMCRunner,
    model::GenericModel,
    sampler::MCSampler,
    tuner::MCTuner,
    pindex::Int,
    vstates::Vector{VariableState},
    outopts::Dict{Symbol, Any}, # Options related to output
    plain::Bool,
    checkin::Bool
  ) = begin
    instance = new()

    instance.runner = runner
    instance.model = model
    instance.sampler = sampler
    instance.tuner = tuner

    instance.pindex = pindex

    instance.vstates = vstates
    initialize!(instance.vstates, pindex, model.vertices[pindex], sampler)

    instance.sstate = sampler_state(instance.vstates[pindex], sampler, tuner)

    augment!(outopts)
    instance.output = initialize_output(instance.vstates[pindex], length(runner.postrange), outopts)

    # TODO: complete inner constructor

    instance
  end
end

# Note: it is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
