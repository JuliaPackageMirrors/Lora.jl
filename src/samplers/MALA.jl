### Abstract MALA state

abstract MALAState <: MCSamplerState

### MALA state subtypes

## UnvMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

type UnvMALAState{N<:Real} <: MALAState
  pstate::ParameterState{Continuous, Univariate, N} # Parameter state used internally by MALA
  driftstep::N # Drift stepsize for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::N
  vmean::N
  pnewgivenold::N
  poldgivennew::N

  function UnvMALAState(
    pstate::ParameterState{Continuous, Univariate, N},
    driftstep::N,
    tune::MCTunerState,
    ratio::N,
    vmean::N,
    pnewgivenold::N,
    poldgivennew::N
  )
    if !isnan(driftstep)
      @assert driftstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

UnvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Univariate, N},
  driftstep::N,
  tune::MCTunerState,
  ratio::N,
  vmean::N,
  pnewgivenold::N,
  poldgivennew::N
) =
  UnvMALAState{N}(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)

UnvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Univariate, N},
  driftstep::N=1.,
  tune::MCTunerState=VanillaMCTune()
) =
  UnvMALAState(pstate, driftstep, tune, NaN, NaN, NaN, NaN)

Base.eltype{N<:Real}(::Type{UnvMALAState{N}}) = N
Base.eltype{N<:Real}(s::UnvMALAState{N}) = N

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

type MuvMALAState{N<:Real} <: MALAState
  pstate::ParameterState{Continuous, Multivariate, N} # Parameter state used internally by MALA
  driftstep::N # Drift stepsize for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::N
  vmean::Vector{N}
  pnewgivenold::N
  poldgivennew::N

  function MuvMALAState(
    pstate::ParameterState{Continuous, Multivariate, N},
    driftstep::N,
    tune::MCTunerState,
    ratio::N,
    vmean::Vector{N},
    pnewgivenold::N,
    poldgivennew::N
  )
    if !isnan(driftstep)
      @assert driftstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

MuvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate, N},
  driftstep::N,
  tune::MCTunerState,
  ratio::N,
  vmean::Vector{N},
  pnewgivenold::N,
  poldgivennew::N
) =
  MuvMALAState{N}(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)

MuvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate, N},
  driftstep::N=1.,
  tune::MCTunerState=VanillaMCTune()
) =
  MuvMALAState(pstate, driftstep, tune, NaN, Array(N, pstate.size), NaN, NaN)

Base.eltype{N<:Real}(::Type{MuvMALAState{N}}) = N
Base.eltype{N<:Real}(s::MuvMALAState{N}) = N

### Metropolis-adjusted Langevin Algorithm (MALA)

immutable MALA <: LMCSampler
  driftstep::Real

  function MALA(driftstep::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)

### Initialize MALA sampler

## Initialize parameter state

function initialize!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate, N},
  vstate::Vector{S},
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
  @assert isfinite(pstate.gradlogtarget) "Gradient of log-target not finite: initial values out of parameter support"
end

function initialize!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate, N},
  vstate::Vector{S},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of parameter support"
end

## Initialize MuvMALAState

sampler_state{N<:Real}(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Univariate, N}) =
  UnvMALAState(generate_empty(pstate), sampler.driftstep, tuner_state(sampler, tuner))

sampler_state{N<:Real}(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate, N}) =
  MuvMALAState(generate_empty(pstate), sampler.driftstep, tuner_state(sampler, tuner))

## Reset parameter state

function reset!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate, N},
  vstate::Vector{S},
  x::N,
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate, vstate)
end

function reset!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate, N},
  vstate::Vector{S},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate, vstate)
end

## Initialize task

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate, N},
  vstate::Vector{S},
  sstate::UnvMALAState{N},
  parameter::Parameter{Continuous, Univariate},
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

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate, N},
  vstate::Vector{S},
  sstate::MuvMALAState{N},
  parameter::Parameter{Continuous, Multivariate},
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
