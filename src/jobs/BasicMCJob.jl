### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  runner::BasicMCRunner
  model::GenericModel # Model of a single parameter
  sampler::MCSampler
  tuner::MCTuner
  index::Int # Index of single parameter in vstates
  vstates::Vector{VariableState} # Vector of variable states ordered according to variables in model.vertices
  pvstates::Vector{VariableState} # Vector of proposed variable states ordered according to variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  output::Union{VariableNState, VariableIOStream} # Output of model's single parameter
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  send::Function
  receive::Function
  reset::Function
  # save::Function
  # close::function
  count::Int # Current number of iterations
  # checkin::Bool # If checkin=true then check validity of job constructors' input arguments, else don't check

  BasicMCJob(
    runner::BasicMCRunner,
    model::GenericModel,
    sampler::MCSampler,
    tuner::MCTuner,
    index::Int,
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

    instance.index = index

    instance.vstates = vstates
    initialize!(instance.vstates, model.vertices[index], sampler, index)

    # TODO: initialize pvstates (stands for proposed variable states)

    instance.sstate = sampler_state(sampler, tuner, instance.vstates[index])

    augment!(outopts)
    instance.output = initialize_output(instance.vstates[index], runner.npoststeps, outopts)

    instance.plain = plain
    if plain
      instance.task = nothing
      instance.send = identity
      # instance.receive = (i::Int)->iterate!(job.heap[i], m[i], s[i], r[i], t[i], identity)
      # instance.reset = (i::Int, x::Vector{Float64})->reset!(job.heap[i], x)
    else
      # instance.task = Task[Task(()->initialize_task!(job.heap[i], m[i], s[i], r[i], t[i])) for i = 1:job.dim]
      instance.send = produce
      # instance.receive = (i::Int)->consume(job.task[i])
      # instance.reset = (i::Int, x::Vector{Float64})->reset(job.task[i], x)
    end

    instance.count = 1

    instance
  end
end

# Note: it is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
