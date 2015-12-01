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

  function MH(symmetric::Bool, logproposal::Union{Function, Void}, randproposal::Function)
    if symmetric && (logproposal != nothing)
      error("If the symmetric field is true, then logproposal is not used in the calculations")
    end
    new(symmetric, logproposal, randproposal)
  end
end

# Metropolis-Hastings sampler (asymmetric proposal)

MH(logproposal::Function, randproposal::Function) = MH(false, logproposal, randproposal)

# Metropolis sampler (symmetric proposal)

MH(randproposal::Function) = MH(true, nothing, randproposal)

# Random-walk Metropolis, i.e. Metropolis with a normal proposal density

MH{N<:Real}(σ::Matrix{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:Real}(σ::Vector{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:Real}(σ::N) = MH(x::N -> rand(Normal(x, σ)))
MH{N<:Real}(::Type{N}=Float64) = MH(x::N -> rand(Normal(x, 1.0)))

### Initialize Metropolis-Hastings sampler

## Initialize parameter state

function initialize!{S<:VariableState}(
  pstate::ContinuousParameterState,
  vstate::Vector{S},
  parameter::ContinuousParameter,
  sampler::MH
)
  parameter.logtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Initial values out of parameter support"
end

## Initialize MHState

sampler_state(sampler::MH, tuner::MCTuner, pstate::ParameterState) = MHState(generate_empty(pstate), tuner_state(tuner))

## Reset parameter state

function reset!{N<:Real, S<:VariableState}(
  pstate::ContinuousUnivariateParameterState,
  vstate::Vector{S},
  x::N,
  parameter::ContinuousUnivariateParameter,
  sampler::MH
)
  pstate.value = x
  parameter.logtarget!(pstate, vstate)
end

function reset!{N<:Real, S<:VariableState}(
  pstate::ContinuousMultivariateParameterState,
  vstate::Vector{S},
  x::Vector{N},
  parameter::ContinuousMultivariateParameter,
  sampler::MH
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate, vstate)
end

## Initialize task

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::ContinuousUnivariateParameterState{N},
  vstate::Vector{S},
  sstate::MHState{ContinuousUnivariateParameterState{N}},
  parameter::ContinuousUnivariateParameter,
  sampler::MH,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, vstate, sstate, parameter, sampler, tuner, range)
  end
end

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::ContinuousMultivariateParameterState{N},
  vstate::Vector{S},
  sstate::MHState{ContinuousMultivariateParameterState{N}},
  parameter::ContinuousMultivariateParameter,
  sampler::MH,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, vstate, sstate, parameter, sampler, tuner, range)
  end
end
