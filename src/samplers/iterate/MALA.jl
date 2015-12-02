function codegen_iterate_mala(job::BasicMCJob, outopts::Dict)
  result::Expr
  update::Vector{Expr}
  noupdate = []
  body = []

  if job.tuner.verbose
    push!(body, :($(:_sstate).tune.proposed += 1))
  end

  push!(body, :($(:_sstate).vmean = $(:_pstate).value+0.5*$(:_sstate).driftstep*$(:_pstate).gradlogtarget))

  vform = variate_form(job.pstate)
  if vform == Univariate
    push!(body, :($(:_sstate).pstate.value = $(:_sstate).vmean+sqrt($(:_sstate).driftstep)*randn()))
  elseif vform == Multivariate
    push!(body, :($(:_sstate).pstate.value = $(:_sstate).vmean+sqrt($(:_sstate).driftstep)*randn($(:_pstate).size)))
  else
    error("Only univariate or multivariate parameter states allowed in MALA code generation")
  end
  push!(body, :($(:_parameter).uptogradlogtarget!($(:_sstate).pstate, $(:_vstate))))

  ### Under development
end
