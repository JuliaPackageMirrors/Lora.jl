function codegen_iterate_mh(job::BasicMCJob, outopts::Dict{Symbol, Any})
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

    push!(body, :(
      if $(job).count <= $(job).range.burnin && mod($(job).count, $(job).tuner.period) == 0
        rate!($(job).sstate.tune)
        println("Burnin iteration $($job.count) of $($job.range.burnin): ", round(100*$(job).sstate.tune.rate, 2), " % acceptance rate")
      end
    ))
  else
    push!(body, :(
      if $(job).sstate.ratio > 0 || ($(job).sstate.ratio > log(rand()))
        $(job).vstate[$(job).vindex].value = copy($(job).sstate.pstate.value)
        $(job).vstate[$(job).vindex].logtarget = copy($(job).sstate.pstate.logtarget)
      end
    ))
  end

  #push!(body, :(println($job.count)))

  @gensym iterate_mh

  quote
    function $iterate_mh()
      $(body...)
    end
  end
end

function codegen_iterate_basic_mcjob(job::BasicMCJob, outopts::Dict{Symbol, Any})
  result::Expr

  if isa(job.sampler, MH)
    result = codegen_iterate_mh(job, outopts)
  end

  result
end
