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

MH{N<:AbstractFloat}(σ::Matrix{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:AbstractFloat}(σ::Vector{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:AbstractFloat}(σ::N) = MH(x::N -> rand(Normal(x, σ)))
MH{N<:AbstractFloat}(::Type{N}=Float64) = MH(x::N -> rand(Normal(x, 1.0)))

### Initialize Metropolis-Hastings sampler

## Initialize variable states

function initialize!(vstate::Vector{VariableState}, parameter::ContinuousParameter, vindex::Int, sampler::MHSampler)
  parameter.logtarget!(vstate[vindex], vstate)
  @assert isfinite(vstate[vindex].logtarget) "Initial values out of model support"
end

## Initialize MHState

sampler_state(sampler::MHSampler, tuner::MCTuner, pstate::ParameterState) =
  MHState(generate_empty(pstate), tuner_state(tuner))

function reset!{N<:AbstractFloat}(
  vstate::Vector{VariableState},
  x::N,
  parameter::ContinuousUnivariateParameter,
  vindex::Int,
  sampler::MHSampler
)
  vstate[vindex].value = x
  parameter.logtarget!(vstate[vindex], vstate)
end

function reset!{N<:AbstractFloat}(
  vstate::Vector{VariableState},
  x::Vector{N},
  parameter::ContinuousMultivariateParameter,
  vindex::Int,
  sampler::MHSampler
)
  vstate[vindex].value = copy(x)
  parameter.logtarget!(vstate[vindex], vstate)
end

function initialize_task!{N<:AbstractFloat}(
  vstate::Vector{VariableState},
  sstate::MHState{ContinuousUnivariateParameterState{N}},
  parameter::ContinuousUnivariateParameter,
  vindex::Int,
  sampler::MHSampler,
  tuner::MCTuner,
  range::BasicMCRange,
  outopts::Dict{Symbol, Any},
  count::Int
)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, x::N -> reset!(vstate, x, parameter, vindex, sampler))

  while true
    iterate!(vstate, sstate, parameter, vindex, sampler, tuner, range, outopts, count, produce)
  end
end

function initialize_task!{N<:AbstractFloat}(
  vstate::Vector{VariableState},
  sstate::MHState{ContinuousMultivariateParameterState{N}},
  parameter::ContinuousMultivariateParameter,
  vindex::Int,
  sampler::MHSampler,
  tuner::MCTuner,
  range::BasicMCRange,
  outopts::Dict{Symbol, Any},
  count::Int
)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, x::Vector{N} -> reset!(vstate, x, parameter, vindex, sampler))

  while true
    iterate!(vstate, sstate, parameter, vindex, sampler, tuner, range, outopts, count, produce)
  end
end

function codegen_iterate_basic_mcjob(
  vstate::Vector{VariableState},
  sstate::MHState,
  parameter::ContinuousParameter,
  vindex::Int,
  sampler::MHSampler,
  tuner::MCTuner,
  range::BasicMCRange,
  outopts::Dict{Symbol, Any},
  count::Int,
  plain::Bool
)
  body = []

  if tuner.verbose
    push!(body, :($(sstate).tune.proposed += 1))
  end

  push!(body, :($(sstate).pstate.value = $(sampler).randproposal($(vstate)[$(vindex)].value)))
  push!(body, :($(parameter).logtarget!($(sstate).pstate, $(vstate))))

  if sampler.symmetric
    push!(body, :($(sstate).ratio = $(sstate).pstate.logtarget-$(vstate)[$(vindex)].logtarget))
  else
    push!(body, :($(sstate).ratio = (
      $(sstate).pstate.logtarget
      +$(sampler).logproposal($(sstate).pstate.value, $(vstate)[$(vindex)].value)
      -$(vstate)[$(vindex)].logtarget
      -$(sampler).logproposal($(vstate)[$(vindex)].value, $(sstate).pstate.value)
    )))
  end

  if tuner.verbose
    push!(body, :(
      if $(sstate).ratio > 0 || ($(sstate).ratio > log(rand()))
        $(vstate)[$(vindex)].value = copy($(sstate).pstate.value)
        $(vstate)[$(vindex)].logtarget = copy($(sstate).pstate.logtarget)

        $(sstate).tune.accepted += 1
      end
    ))

    #push!(body, :(
    #  if $(count) <= $(job).range.burnin && mod($(count), $(tuner).period) == 0
    #    tune!($(sstate).tune, $(tuner))
    #    println("Burnin iteration $(job.count) of $(job.range.burnin): ", round(100*$(sstate).tune.rate, 2), " % acceptance rate")
    #  end
    #end
  else
    push!(body, :(
      if $(sstate).ratio > 0 || ($(sstate).ratio > log(rand()))
        $(vstate)[$(vindex)].value = copy($(sstate).pstate.value)
        $(vstate)[$(vindex)].logtarget = copy($(sstate).pstate.logtarget)
      end
    ))
  end

  if !plain
    push!(body, :(produce()))
  end

  @gensym iterate_basic_mcjob

  quote
    function $iterate_basic_mcjob()
      $(body...)
    end
  end
end
