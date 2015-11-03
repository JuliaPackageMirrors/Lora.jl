function codegen_iterate_basic_mcjob(job::BasicMCJob, outopts::Dict{Symbol, Any})
  result::Expr

  if isa(job.sampler, MH)
    result = codegen_iterate_mh(job, outopts)
  end

  result
end
