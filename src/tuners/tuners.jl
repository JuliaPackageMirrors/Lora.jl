# Tuner state types hold the samplers' temporary output used for tuning the sampler

### MCTunerState

abstract MCTunerState

### BasicMCTune

# BasicMCTune is the most elemental tune type
# It monitors the acceptance rate for the current tuning period
# It stores only the number of accepted and proposed MCMC samples and the observed acceptance rate

type BasicMCTune <: MCTunerState
  accepted::Int # Number of accepted MCMC samples during current tuning period
  proposed::Int # Number of proposed MCMC samples during current tuning period
  rate::Real # Observed acceptance rate over current tuning period

  function BasicMCTune(accepted::Int, proposed::Int, rate::Real)
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(accepted, proposed, rate)
  end
end

BasicMCTune(accepted::Int=0, proposed::Int=0) = BasicMCTune(accepted::Int, proposed::Int, NaN)

reset!(tune::BasicMCTune) = ((tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN))

count!(tune::BasicMCTune) = (tune.accepted += 1)

rate!(tune::BasicMCTune) = (tune.rate = tune.accepted/tune.proposed)

### MCTuner

abstract MCTuner
