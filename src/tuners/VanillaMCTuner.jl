### VanillaMCTune

# VanillaMCTune is the most elemental tune type
# It monitors the acceptance rate for the current tuning period
# It stores only the number of accepted and proposed MCMC samples and the observed acceptance rate

type VanillaMCTune <: MCTunerState
  accepted::Int # Number of accepted MCMC samples during current tuning period
  proposed::Int # Number of proposed MCMC samples during current tuning period
  rate::Real # Observed acceptance rate over current tuning period

  function VanillaMCTune(accepted::Int, proposed::Int, rate::Real)
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(accepted, proposed, rate)
  end
end

VanillaMCTune(accepted::Int=0, proposed::Int=0) = VanillaMCTune(accepted::Int, proposed::Int, NaN)

reset!(tune::VanillaMCTune) = ((tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN))

count!(tune::VanillaMCTune) = (tune.accepted += 1)

rate!(tune::VanillaMCTune) = (tune.rate = tune.accepted/tune.proposed)

### VanillaMCTuner

# VanillaMCTuner is a dummy tuner type in the sense that it does not perform any tuning
# It is used only for determining whether the MCSampler will be verbose

immutable VanillaMCTuner <: MCTuner
  period::Int # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function VanillaMCTuner(period::Int, verbose::Bool)
    @assert period > 0 "Adaptation period should be positive"
    new(period, verbose)
  end
end

VanillaMCTuner(; period::Int=100, verbose::Bool=false) = VanillaMCTuner(period, verbose)
