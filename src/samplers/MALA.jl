### Abstract MALA state

abstract MALAState <: MCSamplerState

### MALA state subtypes

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

type MuvMALAState{N<:Real} <: MALAState
  pstate::BasicContMuvParameterState{N} # Parameter state used internally by MALA
  driftstep::N # Drift step size for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::N
  smean::Vector{N}
  pnewgivenold::N
  poldgivennew::N

  function MuvMALAState(
    pstate::BasicContMuvParameterState{N},
    driftstep::N,
    tune::MCTunerState,
    ratio::N,
    smean::Vector{N},
    pnewgivenold::N,
    poldgivennew::N
  )
    if !isnan(driftstep)
      @assert driftstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, driftstep, tune, ratio, smean, pnewgivenold, poldgivennew)
  end
end

MuvMALAState{N<:Real}(
  pstate::BasicContMuvParameterState{N},
  driftstep::N,
  tune::MCTunerState,
  ratio::N,
  smean::Vector{N},
  pnewgivenold::N,
  poldgivennew::N
) =
  MuvMALAState{N}(pstate, driftstep, tune, ratio, smean, pnewgivenold, poldgivennew)

MuvMALAState{N<:Real}(pstate::BasicContMuvParameterState{N}, tune::MCTunerState=BasicMCTune()) =
  MuvMALAState(pstate, NaN, tune, NaN, Array(N, pstate.size), NaN, NaN)

Base.eltype{N<:Real}(::Type{MuvMALAState{N}}) = N
Base.eltype{N<:Real}(s::MuvMALAState{N}) = N

### Metropolis-adjusted Langevin Algorithm (MALA)

immutable MALA <: LMCSampler
  driftstep::Real

  function MALA(d::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)

### Initialize MALA sampler

## Initialize parameter state

function initialize!{S<:VariableState}(
  pstate::BasicContMuvParameterState,
  vstate::Vector{S},
  parameter::BasicContMuvParameter,
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
  @assert isfinite(pstate.gradlogtarget) "Gradient of log-target not finite: initial values out of parameter support"
end

## Initialize MuvMALAState

sampler_state(sampler::MALA, tuner::MCTuner, pstate::BasicContMuvParameterState) =
  MuvMALAState(generate_empty(pstate), tuner_state(tuner))

## Reset parameter state

function reset!{N<:Real, S<:VariableState}(
  pstate::BasicContMuvParameterState{N},
  vstate::Vector{S},
  x::Vector{N},
  parameter::BasicContMuvParameter,
  sampler::MALA
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate, vstate)
end

## Initialize task

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::BasicContMuvParameterState{N},
  vstate::Vector{S},
  sstate::MuvMALAState{N},
  parameter::BasicContMuvParameter,
  sampler::MALA,
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
