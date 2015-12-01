### Abstract MALA state

abstract MALAState <: MCSamplerState

### MALA state subtypes

## MultivariateMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

type MultivariateMALAState{N<:Real} <: MALAState
  pstate::ContinuousMultivariateParameterState{N} # Parameter state used internally by MALA
  driftstep::N # Drift step size for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::N
  smean::Vector{N}
  pnewgivenold::N
  poldgivennew::N

  function MultivariateMALAState(
    pstate::ContinuousMultivariateParameterState{N},
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

MultivariateMALAState{N<:Real}(
  pstate::ContinuousMultivariateParameterState{N},
  driftstep::N,
  tune::MCTunerState,
  ratio::N,
  smean::Vector{N},
  pnewgivenold::N,
  poldgivennew::N
) =
  MultivariateMALAState{N}(pstate, driftstep, tune, ratio, smean, pnewgivenold, poldgivennew)

MultivariateMALAState{N<:Real}(pstate::ContinuousMultivariateParameterState{N}, tune::MCTunerState=BasicMCTune()) =
  MultivariateMALAState(pstate, NaN, tune, NaN, Array(N, pstate.size), NaN, NaN)

Base.eltype{N<:Real}(::Type{MultivariateMALAState{N}}) = N
Base.eltype{N<:Real}(s::MultivariateMALAState{N}) = N

### Metropolis-adjusted Langevin Algorithm (MALA)

immutable MALA <: LMCSampler
  driftstep::Real

  function MALA(d::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)
