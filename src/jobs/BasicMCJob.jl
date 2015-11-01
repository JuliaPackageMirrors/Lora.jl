### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  model::GenericModel # Model of single parameter
  vindex::Int # Index of single parameter in model.vertices
  sampler::MCSampler
  tuner::MCTuner
  range::BasicMCRange
  vstate::Vector{VariableState} # Vector of variable states ordered according to variables in model.vertices
  sstate::MCSamplerState # Internal state of MCSampler
  output::Union{VariableNState, VariableIOStream} # Output of model's single parameter
  count::Int # Current number of iterations
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  send::Function
  receive::Function
  reset::Function
  save::Function
  close::Function
  # checkin::Bool # If checkin=true then check validity of job constructors' input arguments, else don't check

  BasicMCJob(
    model::GenericModel,
    vindex::Int,
    sampler::MCSampler,
    tuner::MCTuner,
    range::BasicMCRange,
    vstate::Vector{VariableState},
    outopts::Dict{Symbol, Any}, # Options related to output
    plain::Bool,
    checkin::Bool
  ) = begin
    instance = new()

    instance.model = model
    instance.vindex = vindex

    instance.sampler = sampler
    instance.tuner = tuner
    instance.range = range

    instance.vstate = vstate
    initialize!(instance.vstate, model.vertices[vindex], vindex, sampler)

    instance.sstate = sampler_state(sampler, tuner, instance.vstate[vindex])

    augment!(outopts)
    instance.output = initialize_output(instance.vstate[vindex], range.npoststeps, outopts)

    instance.count = 0

    instance.plain = plain
    if plain
      instance.task = nothing
      instance.send = identity
      instance.receive = () -> iterate!(
        instance.vstate,
        instance.sstate,
        model.vertices[vindex],
        vindex,
        sampler,
        tuner,
        range,
        outopts,
        instance.count,
        identity
      )
      instance.reset = x::Vector -> reset!(instance.vstate, x, model.vertices[vindex], vindex, sampler)
    else
      instance.task = Task(() -> initialize_task!(
        instance.vstate, instance.sstate, parameter, vindex, sampler, tuner, range, outopts, count)
      )
      instance.send = produce
      instance.receive = () -> consume(instance.task)
      instance.reset = x::Vector -> reset(instancd.task, x)
    end

    if outopts[:destination] == :nstate
      instance.save = (state::ParameterState, i::Int) -> instance.output.copy(state, i)
      instance.close = () -> ()
    elseif outopts[:destination] == :iostream
      instance.save = (state::ParameterState, i::Int) -> instance.output.write(state)
      instance.close = () -> close(instance.output)
    end

    instance
  end
end

# It is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
# In that case the iterate!() function will take a second variable (transformation) as input argument
