### MHState

# MHState holds the internal state ("local variables") of the Metropolis-Hastings sampler

type MHState{S<:ParameterState} <: MCSamplerState
  pstate::S # Parameter state used internally by MH
  tune::MCTunerState
  ratio::Real # Acceptance ratio

  function MHState(pstate::S, tune::MCTunerState, ratio::Real)
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio)
  end
end

MHState{S<:ParameterState}(pstate::S, tune::MCTunerState, ratio::Real) = MHState{S}(pstate, tune, ratio)

MHState{S<:ParameterState}(pstate::S, tune::MCTunerState=BasicMCTune()) = MHState(pstate, tune, NaN)

Base.eltype{S<:ParameterState}(::Type{MHState{S}}) = S
Base.eltype{S<:ParameterState}(s::MHState{S}) = S

### Metropolis-Hastings (MH) sampler

# In its most general case it accommodates an asymmetric proposal density
# For symetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing

immutable MH <: MHSampler
  symmetric::Bool # If symmetric=true then the proposal density is symmetric, else it is asymmetric
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
MH(σ::AbstractFloat) = MH(x::AbstractFloat -> rand(Normal(x, σ)))
MH() = MH(x::AbstractFloat -> rand(Normal(x, 1.0)))

### Initialize Metropolis-Hastings sampler

## Initialize variable states

function initialize!(vstates::Vector{VariableState}, parameter::ContinuousParameter, sampler::MHSampler, index::Int)
  parameter.logtarget!(vstates, index)
  @assert isfinite(vstates[index].logtarget) "Initial values out of model support"
end

## Initialize MHState

sampler_state(sampler::MHSampler, tuner::MCTuner, pstate::ParameterState) =
  MHState(generate_empty(pstate), tuner_state(tuner))

function reset!{N<:AbstractFloat}(
  vstates::Vector{VariableState},
  x::N,
  parameter::ContinuousUnivariateParameterState{N},
  sampler::MHSampler,
  index::Int
)
  vstates[index].value = x
  parameter.logtarget!(vstates, index)
end

function reset!{N<:AbstractFloat}(
  vstates::Vector{VariableState},
  x::Vector{N},
  parameter::ContinuousMultivariateParameterState{N},
  sampler::MHSampler,
  index::Int
)
  vstates[index].value = copy(x)
  parameter.logtarget!(vstates, index)
end

# function initialize_task!{N<:AbstractFloat}(
#   vstates::Vector{VariableState},
#   sstate::MHState,
#   range::BasicMCRange,
#   parameter::ContinuousParameter,
#   sampler::MHSampler,
#   tuner::MCTuner,
#   index::Int
# )
#   # Hook inside Task to allow remote resetting
#   task_local_storage(:reset, (x::Vector{N})->reset!(vstates, x, parameter, sampler, index))
#
#   while true
#     iterate!(vstates, pvstates, sstate, range, parameter, sampler, tuner, index, produce)
#   end
# end
#
# function iterate!(
#   vstates::Vector{VariableState},
#   pvstates::Vector{VariableState},
#   sstate::MHState,
#   range::BasicMCRange,
#   parameter::ContinuousParameter,
#   sampler::MHSampler,
#   tuner::MCTuner,
#   index::Int,
#   send::Function
# )
#   if tuner.verbose
#     sstate.tune.proposed += 1
#   end
#
#   pvstates[index].value = sampler.randproposal(vstates[index].value)
#   parameter.logtarget!(pvstates, index)
#
#   if sampler.symmetric
#     sstate.ratio = pvstates.logtarget-vstates.logtarget
#   else
#     heap.ratio = (
#       pvstates.logtarget
#       +sampler.logproposal(pvstates.value, vstates.value)
#       -vstates.logtarget
#       -sampler.logproposal(vstates.value, pvstates.value
#     )
#   end
#
#   if sstate.ratio > 0 || (sstate.ratio > log(rand()))
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
