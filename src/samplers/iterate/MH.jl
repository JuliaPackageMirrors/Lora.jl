function codegen_iterate_mh(job::BasicMCJob, outopts::Dict{Symbol, Any})
  body = []

  if job.tuner.verbose
    push!(body, :($(:_sstate).tune.proposed += 1))
  end

  push!(body, :($(:_sstate).pstate.value = $(:_sampler).randproposal($(:_pstate).value)))
  push!(body, :($(:_parameter).logtarget!($(:_sstate).pstate, $(:_vstate))))

  if job.sampler.symmetric
    push!(body, :($(:_sstate).ratio = $(:_sstate).pstate.logtarget-$(:_pstate).logtarget))
  else
    push!(body, :($(:_sstate).ratio = (
      $(:_sstate).pstate.logtarget
      +$(:_sampler).logproposal($(:_sstate).pstate.value, $(:_pstate).value)
      -$(:_pstate).logtarget
      -$(:_sampler).logproposal($(:_pstate).value, $(:_sstate).pstate.value)
    )))
  end

  if job.tuner.verbose
    push!(body, :(
      if $(:_sstate).ratio > 0 || ($(:_sstate).ratio > log(rand()))
        $(:_pstate).value = copy($(:_sstate).pstate.value)
        $(:_pstate).logtarget = copy($(:_sstate).pstate.logtarget)

        $(:_sstate).tune.accepted += 1
      end
    ))

    push!(body, :(
      if $(:_count) <= $(:_range).burnin && mod($(:_count), $(:_tuner).period) == 0
        rate!($(:_sstate).tune)
        println(
          "Burnin iteration ",
          $(:_count),
          " of ",
          $(:_range).burnin,
          ": ",
          round(100*$(:_sstate).tune.rate, 2),
          " % acceptance rate"
        )
      end
    ))
  else
    push!(body, :(
      if $(:_sstate).ratio > 0 || ($(:_sstate).ratio > log(rand()))
        $(:_pstate).value = copy($(:_sstate).pstate.value)
        $(:_pstate).logtarget = copy($(:_sstate).pstate.logtarget)
      end
    ))
  end

  # push!(body, :(println($(:_count))))

  @gensym iterate_mh

  quote
    function $iterate_mh(
      _pstate::ContinuousMultivariateParameterState,
      _vstate::Vector{VariableState},
      _sstate::MHState,
      _parameter::ContinuousMultivariateParameter,
      _sampler::MH,
      _tuner::MCTuner,
      _range::BasicMCRange,
      _count::Int,
      _plain::Bool
    )
      $(body...)
    end
  end
end
