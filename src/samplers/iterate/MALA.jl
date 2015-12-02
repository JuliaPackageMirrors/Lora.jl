function codegen_iterate_mala(job::BasicMCJob, outopts::Dict)
  result::Expr
  update::Vector{Expr}
  noupdate = []
  body = []

  if job.tuner.verbose
    push!(body, :($(:_sstate).tune.proposed += 1))
  end

  ### Under development
end
