function codegen_iterate_mala(job::BasicMCJob, outopts::Dict)
  result::Expr
  update::Vector{Expr}
  noupdate = []
  body = []

  stepsize = isa(job.tuner, AcceptanceRateMCTuner) ? :($(:_sstate).tune.step) : :($(:_sstate).driftstep)

  if job.tuner.verbose
    push!(body, :($(:_sstate).tune.proposed += 1))
  end

  push!(body, :($(:_sstate).vmean = $(:_pstate).value+0.5*$(stepsize)*$(:_pstate).gradlogtarget))

  vform = variate_form(job.pstate)
  if (vform != Univariate) && (vform != Multivariate)
    error("Only univariate or multivariate parameter states allowed in MALA code generation")
  end

  if vform == Univariate
    push!(body, :($(:_sstate).pstate.value = $(:_sstate).vmean+sqrt($(stepsize))*randn()))
  elseif vform == Multivariate
    push!(body, :($(:_sstate).pstate.value = $(:_sstate).vmean+sqrt($(stepsize))*randn($(:_pstate).size)))
  end

  push!(body, :($(:_parameter).uptogradlogtarget!($(:_sstate).pstate, $(:_vstate))))

  if vform == Univariate
    push!(body, :($(:_sstate).pnewgivenold =
      -0.5*(abs2($(:_sstate).vmean-$(:_sstate).pstate.value)/$(stepsize)+log(2*pi*$(stepsize)))
    ))
  elseif vform == Multivariate
    push!(body, :($(:_sstate).pnewgivenold =
      sum(-0.5*(abs2($(:_sstate).vmean-$(:_sstate).pstate.value)/$(stepsize)+log(2*pi*$(stepsize))))
    ))
  end

  push!(body, :(
    $(:_sstate).vmean = $(:_sstate).pstate.value+0.5*$(stepsize)*$(:_sstate).pstate.gradlogtarget
  ))

  if vform == Univariate
    push!(body, :($(:_sstate).pnewgivenold =
      -0.5*(abs2($(:_sstate).vmean-$(:_pstate).value)/$(stepsize)+log(2*pi*$(stepsize)))
    ))
  elseif vform == Multivariate
    push!(body, :($(:_sstate).pnewgivenold =
      sum(-0.5*(abs2($(:_sstate).vmean-$(:_pstate).value)/$(stepsize)+log(2*pi*$(stepsize))))
    ))
  end

  push!(body, :(
    $(:_sstate).ratio = $(:_sstate).pstate.logtarget+$(:_sstate).poldgivennew-$(:_pstate).logtarget-$(:_sstate).pnewgivenold
  ))

  update = [
    :($(:_pstate).value = copy($(:_sstate).pstate.value)),
    :($(:_pstate).logtarget = copy($(:_sstate).pstate.logtarget)),
    :($(:_pstate).gradlogtarget = copy($(:_sstate).pstate.gradlogtarget))
  ]
  if in(:loglikelihood, outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :($(:_pstate).loglikelihood = copy($(:_sstate).pstate.loglikelihood)))
  end
  if in(:logprior, outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :($(:_pstate).logprior = copy($(:_sstate).pstate.logprior)))
  end
  if in(:gradloglikelihood, outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :($(:_pstate).gradloglikelihood = copy($(:_sstate).pstate.gradloglikelihood)))
  end
  if in(:gradlogprior, outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :($(:_pstate).gradlogprior = copy($(:_sstate).pstate.gradlogprior)))
  end
  if in(:accept, outopts[:diagnostics])
    push!(update, :($(:_pstate).diagnosticvalues[1] = true))
    push!(noupdate, :($(:_pstate).diagnosticvalues[1] = false))
  end
  if job.tuner.verbose
    push!(update, :($(:_sstate).tune.accepted += 1))
  end

  push!(body, Expr(:if, :($(:_sstate).ratio > 0 || ($(:_sstate).ratio > log(rand()))), Expr(:block, update...), noupdate...))











  if job.tuner.verbose
    push!(body, :(
      if $(:_sstate).tune.proposed <= $(:_range).burnin && mod($(:_sstate).tune.proposed, $(:_tuner).period) == 0
        rate!($(:_sstate).tune)
        println(
          "Burnin iteration ",
          $(:_sstate).tune.proposed,
          " of ",
          $(:_range).burnin,
          ": ",
          round(100*$(:_sstate).tune.rate, 2),
          " % acceptance rate"
        )
      end
    ))
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym iterate_mala

  result = quote
    function $iterate_mala{S<:VariableState}(
      _pstate::$(typeof(job.pstate)),
      _vstate::Vector{S},
      _sstate::$(typeof(job.sstate)),
      _parameter::$(typeof(job.parameter)),
      _sampler::MALA,
      _tuner::MCTuner,
      _range::BasicMCRange
    )
      $(body...)
    end
  end

  result
end
