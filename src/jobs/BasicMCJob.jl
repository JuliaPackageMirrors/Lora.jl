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
  iterate!::Function
  receive!::Function
  reset!::Function
  save!::Function
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

    instance.count = 1

    instance.plain = plain

    if plain
      instance.task = nothing
      instance.send = () -> ()
      instance.iterate! = eval(codegen_iterate_basic_mcjob(instance, outopts))
      # instance.receive! = () -> iterate!(
      #   instance.vstate,
      #   instance.sstate,
      #   model.vertices[vindex],
      #   vindex,
      #   sampler,
      #   tuner,
      #   range,
      #   outopts,
      #   instance.count,
      #   () -> ()
      # )
      instance.reset! = x::Vector -> reset!(instance.vstate, x, model.vertices[vindex], vindex, sampler)
    else
      instance.task = Task(() -> initialize_task!(
        instance.vstate, instance.sstate, parameter, vindex, sampler, tuner, range, outopts, count)
      )
      instance.send = produce
      instance.iterate! = () -> consume(instance.task)
      # instance.receive! = () -> consume(instance.task)
      instance.reset! = x::Vector -> reset(instance.task, x)
    end

    if outopts[:destination] == :nstate
      instance.save! = (i::Int) -> instance.output.copy(instance.vstate[vindex], i)
      instance.close = () -> ()
    elseif outopts[:destination] == :iostream
      instance.save! = (i::Int) -> instance.output.write(instance.vstate[vindex])
      instance.close = () -> close(instance.output)
    end

    instance
  end
end

function codegen_iterate_basic_mcjob(job::BasicMCJob, outopts::Dict{Symbol, Any})
  body = []

  if job.tuner.verbose
    push!(body, :($(job).sstate.tune.proposed += 1))
  end

  push!(body, :($(job).sstate.pstate.value = $(job).sampler.randproposal($(job).vstate[$(job).vindex].value)))
  push!(body, :($(job).model.vertices[$(job).vindex].logtarget!($(job).sstate.pstate, $(job).vstate)))

  if job.sampler.symmetric
    push!(body, :($(job).sstate.ratio = $(job).sstate.pstate.logtarget-$(job).vstate[$(job).vindex].logtarget))
  else
    push!(body, :($(job).sstate.ratio = (
      $(job).sstate.pstate.logtarget
      +$(job).sampler.logproposal($(job).sstate.pstate.value, $(job).vstate[$(job).vindex].value)
      -$(job).vstate[$(job).vindex].logtarget
      -$(job).sampler.logproposal($(job).vstate[$(job).vindex].value, $(job).sstate.pstate.value)
    )))
  end

  if job.tuner.verbose
    push!(body, :(
      if $(job).sstate.ratio > 0 || ($(job).sstate.ratio > log(rand()))
        $(job).vstate[$(job).vindex].value = copy($(job).sstate.pstate.value)
        $(job).vstate[$(job).vindex].logtarget = copy($(job).sstate.pstate.logtarget)

        $(job).sstate.tune.accepted += 1
      end
    ))

    #push!(body, :(
    #  if $(job).count <= $(job).range.burnin && mod($(job).count, $(job).tuner.period) == 0
    #    tune!($(job).sstate.tune, $(job).tuner)
    #    println("Burnin iteration $(job.count) of $(job.range.burnin): ", round(100*$(job).sstate.tune.rate, 2), " % acceptance rate")
    #  end
    #end
  else
    push!(body, :(
      if $(job).sstate.ratio > 0 || ($(job).sstate.ratio > log(rand()))
        $(job).vstate[$(job).vindex].value = copy($(job).sstate.pstate.value)
        $(job).vstate[$(job).vindex].logtarget = copy($(job).sstate.pstate.logtarget)
      end
    ))
  end

  @gensym iterate_basic_mcjob

  quote
    function $iterate_basic_mcjob()
      $(body...)
    end
  end
end

# It is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
# In that case the iterate!() function will take a second variable (transformation) as input argument

function Base.run(job::BasicMCJob)
  for i in 1:job.range.nsteps
    job.iterate!()

    if in(i, job.range.postrange)
      job.save!(job.count)

      job.count += 1
    end
  end

  job.close()

  job.output
end
