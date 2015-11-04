function codegen_iterate_mh(job::BasicMCJob, outopts::Dict{Symbol, Any})
  result::Expr
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

  if isa(job.pstate, ContinuousUnivariateParameterState) &&
    isa(job.sstate.pstate, ContinuousUnivariateParameterState) &&
    isa(job.parameter, ContinuousUnivariateParameter)
    result = quote
      function $iterate_mh{N<:AbstractFloat}(
        _pstate::ContinuousUnivariateParameterState{N},
        _vstate::Vector{VariableState},
        _sstate::MHState{ContinuousUnivariateParameterState{N}},
        _parameter::ContinuousUnivariateParameter,
        _sampler::MH,
        _tuner::MCTuner,
        _range::BasicMCRange,
        _count::Int,
        _plain::Bool
      )
        $(body...)
      end
    end
  elseif isa(job.pstate, ContinuousMultivariateParameterState) &&
    isa(job.sstate.pstate, ContinuousMultivariateParameterState) &&
    isa(job.parameter, ContinuousMultivariateParameter)
    result = quote
      function $iterate_mh{N<:AbstractFloat}(
        _pstate::ContinuousMultivariateParameterState{N},
        _vstate::Vector{VariableState},
        _sstate::MHState{ContinuousMultivariateParameterState{N}},
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
    else
      error("It is not possible to define MH iterate!() for given job")
  end

  result
end
