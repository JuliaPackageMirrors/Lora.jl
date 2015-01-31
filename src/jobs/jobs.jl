### Generic MCJob type is used for Monte Carlo jobs based on coroutines or not 

type MCJob{T, R<:MCRunner}      # T is a symbol for job type, :plain or :task
  model::Vector{MCModel}
  sampler::Vector{MCSampler}
  runner::Vector{R}
  tuner::Vector{MCTuner}

  stash::Vector{MCStash}
  dim::Int
  task::Vector{Task}  # will be empty if plain job

  function MCJob(m, s, r, t)
    vs = length(m)
    @assert vs == length(s) == length(r) == length(t) "Number of models, samplers, runners and tuners not equal."

    stash = MCStash[initialize_stash(m[i], s[i], r[i], t[i]) for i = 1:vs]
    
    if T == :task
      ts = Task[Task(()->initialize_task!(stash[i], m[i], s[i], r[i], t[i])) for i = 1:vs]
      return new(m, s, r, t, stash, vs, ts)
    elseif T == :plain
      return new(m, s, r, t, stash, vs, Task[])
    else
      error("Unknow job type : $T, try :plain or :task")
    end
  end
end

typealias MCPlainJob{R} MCJob{:plain, R}
typealias MCTaskJob{R}  MCJob{:task, R}

### Generic constructor
function MCJob(T, 
               m::MCModel, 
               s::MCSampler=MH(),
               r::MCRunner=SerialMC(nsteps=100), 
               t::MCTuner=VanillaMCTuner())
  rtyp = typeof(r)
  ls   = map(x -> isa(x, AbstractArray) ? length(x) : 1, [m, s, r, t])
  maxn = maximum(ls)

  all((ls .== 1) | (ls .== maxn)) || error("incompatible size of arguments $(unique(ls))")

  ms =   MCModel[ ls[1]==1 ? deepcopy(m) : m[i] for i in 1:maxn ]  # not sure a deepcopy is needed here
  ss = MCSampler[ ls[2]==1 ?          s  : s[i] for i in 1:maxn ]
  rs =  MCRunner[ ls[3]==1 ?          r  : r[i] for i in 1:maxn ]
  ts =   MCTuner[ ls[4]==1 ?          t  : t[i] for i in 1:maxn ]

  MCJob{T,rtyp}(ms, ss, rs, ts)
end


send(    job::MCPlainJob)       = nothing   
receive( job::MCPlainJob, i)    = iterate!(job.stash[i], 
                                           job.model[i], 
                                           job.sampler[i], 
                                           job.runner[i], 
                                           job.tuner[i], 
                                           identity)

reset(   job::MCPlainJob, i, x) = reset!(job.stash[i], x)

send(     job::MCTaskJob)       = produce()
receive(  job::MCTaskJob, i)    = consume(job.task[i])
reset(    job::MCTaskJob, i, x) = reset(job.task[i], x)

### Functions for running jobs
# each runner defines a run(job::MCJob{Symbol, runnertype}) method
# here is only a shortcut method run(type::Symbol, m, s, r, t) that 
# creates the MCJob and calls run(job)

run(T::Symbol, args...) = run( _MCJob(T, args...) )
run(args...) = run( _MCJob(:task, args...) )

### Functions for resuming jobs

# function resume!(j::MCJob, c::MCChain; nsteps::Int=100)
#   if j.dim == 1
#     if isa(j.runner[1], SerialMC)
#       resume!(j.model[1], j.sampler[1], j.runner[1], c, j.tuner[1], j.jobtype; nsteps=nsteps)
#     end
#   end
# end

# function resume(j::MCJob, c::MCChain; nsteps::Int=100)
#   if j.dim == 1
#     if isa(j.runner[1], SerialMC)
#       resume!(deepcopy(j.model)[1], j.sampler[1], j.runner[1], c, j.tuner[1], j.jobtype; nsteps=nsteps)
#     end
#   end
# end
