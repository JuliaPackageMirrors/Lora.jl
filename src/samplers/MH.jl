### MHState

# MHState holds the internal state ("local variables") of the Metropolis-Hastings sampler

type MHState{S<:ContinuousParameterState} <: MCSamplerState
  pstate::S # Parameter state used internally by MH
  tune::MCTune
  count::Int # Current number of iterations
  ratio::Float64 # Acceptance ratio

  function MHState(pstate::S, tune::MCTune, count::Int, ratio::Float64)
    @assert count >= 0 "Number of elapsed MCMC iterations should be non-negative"
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, count, ratio)
  end
end

MHState{S<:ContinuousParameterState}(pstate::S, tune::MCTune, count::Int, ratio::Float64) =
  MHState{S}(pstate, tune, count, ratio)

MHState{S<:ContinuousParameterState}(pstate::S, tune::MCTune=BasicMCTune()) = MHState(pstate, tune, 0, NaN)

Base.eltype{S<:ContinuousParameterState}(::Type{MHState{S}}) = S
Base.eltype{S<:ContinuousParameterState}(s::MHState{S}) = S

### Metropolis-Hastings (MH) sampler

# In its most general case it accommodates an asymmetric proposal density
# For symetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing

immutable MH <: MHSampler
  symmetric::Bool # If the proposal density is symmetric, then symmetric=true, otherwise symmetric=false
  logproposal::Union{Function, Void} # logpdf of asymmetric proposal. For symmetric proposals, logproposal=nothing
  randproposal::Function # random sampling from proposal density

  function MH(s::Bool, l::Union{Function, Void}, r::Function)
    if s && (l != nothing)
      error("If the symmetric field is true, then logproposal is not used in the calculations")
    end
    new(s, l, r)
  end
end

MH(l::Function, r::Function) = MH(false, l, r) # Metropolis-Hastings sampler (asymmetric proposal)
MH(r::Function) = MH(true, nothing, r) # Metropolis sampler (symmetric proposal)

# Random-walk Metropolis, i.e. Metropolis with a normal proposal density
MH(σ::Matrix{AbstractFloat}) = MH(x::Vector{AbstractFloat} -> rand(MvNormal(x, σ)))
MH(σ::Vector{AbstractFloat}) = MH(x::Vector{AbstractFloat} -> rand(MvNormal(x, σ)))
MH(σ::AbstractFloat) = MH(x::Vector{AbstractFloat} -> rand(Normal(x, σ)))
MH() = MH(x::Vector{AbstractFloat} -> rand(Normal(x, 1.0)))

# ### Initialize Metropolis-Hastings sampler
#
# function initialize_heap(m::MCModel, s::MH, r::MCRunner, t::MCTuner)
#   heap::MHHeap = MHHeap(m.size)
#
#   heap.instate.current = MCBaseSample(copy(m.init))
#   logtarget!(heap.instate.current, m.eval)
#   @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."
#
#   heap.tune = VanillaMCTune()
#
#   heap.count = 1
#
#   heap
# end
#
# function reset!(heap::MHHeap, x::Vector{Float64}, m::MCModel)
#   heap.instate.current = MCBaseSample(copy(x))
#   logtarget!(heap.instate.current, m.eval)
# end
#
# function initialize_task!(heap::MHHeap, m::MCModel, s::MH, r::MCRunner, t::MCTuner)
#   # Hook inside Task to allow remote resetting
#   task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x, m))
#
#   while true
#     iterate!(heap, m, s, r, t, produce)
#   end
# end
#
# ### Perform iteration for Metropolis-Hastings sampler
#
# function iterate!(heap::MHHeap, m::MCModel, s::MH, r::MCRunner, t::MCTuner, send::Function)
#   if t.verbose
#     heap.tune.proposed += 1
#   end
#
#   heap.instate.successive = MCBaseSample(s.randproposal(heap.instate.current.sample))
#   logtarget!(heap.instate.successive, m.eval)
#
#   if s.symmetric
#     heap.ratio = heap.instate.successive.logtarget-heap.instate.current.logtarget
#   else
#     heap.ratio = (heap.instate.successive.logtarget
#       +s.logproposal(heap.instate.successive.sample, heap.instate.current.sample)
#       -heap.instate.current.logtarget
#       -s.logproposal(heap.instate.current.sample, heap.instate.successive.sample)
#     )
#   end
#   if heap.ratio > 0 || (heap.ratio > log(rand()))
#     heap.outstate = MCState(heap.instate.successive, heap.instate.current, Dict{Any, Any}("accept" => true))
#     heap.instate.current = deepcopy(heap.instate.successive)
#
#     if t.verbose
#       heap.tune.accepted += 1
#     end
#   else
#     heap.outstate = MCState(heap.instate.current, heap.instate.current, Dict{Any, Any}("accept" => false))
#   end
#
#   if t.verbose && heap.count <= r.burnin && mod(heap.count, t.period) == 0
#     rate!(heap.tune)
#     println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
#   end
#
#   heap.count += 1
#
#   send(heap.outstate)
# end
