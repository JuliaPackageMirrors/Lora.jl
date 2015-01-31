### Serial "ordinary" Monte Carlo runner

immutable SerialMCBaseRunner <: SerialMCRunner
  burnin::Int
  thinning::Int
  nsteps::Int
  r::Range{Int}
  storegradlogtarget::Bool # Indicates whether to save the gradient of the log-target in the cases it is available

  function SerialMCBaseRunner(r::Range{Int}, s::Bool=false)
    burnin = first(r)-1
    thinning = r.step
    nsteps = last(r)
    @assert burnin >= 0 "Number of burn-in iterations should be non-negative."
    @assert thinning >= 1 "Thinning should be >= 1."
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations."
    new(burnin, thinning, nsteps, r, s)
  end
end

SerialMCBaseRunner(r::Range1{Int}, s::Bool=false) = SerialMCBaseRunner(first(r):1:last(r), s)

SerialMCBaseRunner(; burnin::Int=0, thinning::Int=1, nsteps::Int=100, storegradlogtarget::Bool=false) =
  SerialMCBaseRunner((burnin+1):thinning:nsteps, storegradlogtarget)

typealias SerialMC SerialMCBaseRunner

# function run(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), job::Symbol=:task)
function run(mcjob::Union(MCTaskJob{SerialMC}, MCPlainJob{SerialMC}))
  # mcjob = MCJob(m, s, r; tuner=t, job=job)
  r = mcjob.runner[1]  # The runner is unique, so we'll take the first one

  chains = MCChain[]
  for ijob in 1:mcjob.dim
    tic()
    m = mcjob.model[ijob]
    s = mcjob.sampler[ijob]
    t = mcjob.tuner[ijob]

    # Pre-allocation for storing results
    mcchain::MCChain = MCChain(m.size, length(r.r); storegradlogtarget=r.storegradlogtarget)
    ds = Dict{Any, Any}("step" => collect(r.r))

    # Sampling loop
    i::Int = 1
    for j in 1:r.nsteps
      mcstate = receive(mcjob, ijob)
      if in(j, r.r)
        mcchain.samples[i, :] = mcstate.successive.sample
        mcchain.logtargets[i] = mcstate.successive.logtarget

        if r.storegradlogtarget
          mcchain.gradlogtargets[i, :] = mcstate.successive.gradlogtarget
        end

        # Save diagnostics
        for (k,v) in mcstate.diagnostics
          # If diagnostics name not seen before, create column
          if !haskey(ds, k)
            ds[k] = Array(typeof(v), length(ds["step"]))          
          end
          
          ds[k][i] = v
        end

        i += 1
      end
    end

    mcchain.diagnostics, mcchain.runtime = ds, toq()
    push!(chains, mcchain)
  end
  chains
end

function resume(mcjob::Union(MCTaskJob{SerialMC}, MCPlainJob{SerialMC}) ,
                cs::Vector{MCChain}, 
                nsteps::Int=100;
                keep=true)

  for ijob in 1:mcjob.dim
    rg = 1:mcjob.runner[ijob].thinning:nsteps
    mcjob.runner[ijob] = SerialMC(rg, mcjob.runner[ijob].storegradlogtarget)
    mcjob.model[ijob]  = deepcopy(mcjob.model[ijob])        # a copy is made to preserve init values
    # TODO : we should have a separate init vector in the job description, having to make a copy is not optimal
    # the init value in model would only be a hint used if no value is proposed when calling run
    mcjob.model[ijob].init = vec(cs[ijob].samples[end, :])    # start from where we left
  end

  ncs = run(mcjob)

  if keep # if keep = true we add new chains to existing ones
    for ijob in 1:mcjob.dim
      ncs[ijob].samples        = vcat(cs[ijob].samples        , ncs[ijob].samples)
      ncs[ijob].logtargets     = vcat(cs[ijob].logtargets     , ncs[ijob].logtargets)
      ncs[ijob].gradlogtargets = vcat(cs[ijob].gradlogtargets , ncs[ijob].gradlogtargets)

      for k in keys(ncs[ijob].diagnostics)
        ncs[ijob].diagnostics[k] = vcat(cs[ijob].diagnostics[k], ncs[ijob].diagnostics[k]) 
      end
      ncs[ijob].runtime = cs[ijob].runtime + ncs[ijob].runtime 
    end
  end

  ncs
end

# function resume!(m::MCModel, s::MCSampler, r::SerialMC, c::MCChain, t::MCTuner=VanillaMCTuner(), j::Symbol=:task;
#   nsteps::Int=100)
#   m.init = vec(c.samples[end, :])
#   mcrunner::SerialMC = SerialMC(burnin=0, thinning=r.thinning, nsteps=nsteps, storegradlogtarget=r.storegradlogtarget)
#   run(m, s, mcrunner, t, j)
# end

# resume(m::MCModel, s::MCSampler, r::SerialMC, c::MCChain, t::MCTuner=VanillaMCTuner(), j::Symbol=:task;
#   nsteps::Int=100) =
#   resume!(deepcopy(m), s, r, c, t, j; nsteps=nsteps)
