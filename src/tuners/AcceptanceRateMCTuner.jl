### Auxiliary functions used as scores for penalising deviation of observed from target acceptance rate

## logistic_rate_score

# logistic_rate_score allows to scale the acceptance rate by a factor ranging from 0 to 2
# In other words, it allows to nearly eliminate or double the rate depending on its observed value

logistic_rate_score(x::Number, k::Number=7.) = logistic(x, 2., k, 0., 0.)

### AcceptanceRateMCTune

# AcceptanceRateMCTune holds the tuning-related local variables of a MCSampler that uses the AcceptanceRateMCTuner

type AcceptanceRateMCTune <: MCTune
  step::Float64 # Stepsize of MCMC iteration (for ex leapfrog or drift stepsize)
  accepted::Int # Number of accepted MCMC samples during current tuning period
  proposed::Int # Number of proposed MCMC samples during current tuning period
  rate::Float64 # Observed acceptance rate over current tuning period

  function AcceptanceRateMCTune(step::Float64, accepted::Int, proposed::Int, rate::Float64)
    @assert step > 0 "Stepsize of MCMC iteration should be positive"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(step, accepted, proposed, rate)
  end
end

AcceptanceRateMCTune(step::Float64, accepted::Int=0, proposed::Int=0) =
  AcceptanceRateMCTune(step, accepted, proposed, NaN)

reset!(tune::AcceptanceRateMCTune) = ((tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN))

count!(tune::AcceptanceRateMCTune) = (tune.accepted += 1)

rate!(tune::AcceptanceRateMCTune) = (tune.rate = tune.accepted/tune.proposed)

### AcceptanceRateMCTuner

# AcceptanceRateMCTuner tunes empirically on the basis of the discrepancy between observed and target acceptance rate
# This discrepancy is pernalised via a score function set by the user
# The default score function is a stretched logistic map

immutable AcceptanceRateMCTuner <: MCTuner
  targetrate::Float64 # Target acceptance rate
  score::Function # Score function for penalising discrepancy between observed and target acceptance rate
  period::Int # Tuning period over which acceptance rate is computed
  verbose::Bool # If the tuner is verbose then verbose=true, whereas if the tuner is silent then verbose=false

  function AcceptanceRateMCTuner(targetrate::Float64, score::Function, period::Int, verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert period > 0 "Tuning period should be positive"
    new(targetrate, score, period, verbose)
  end
end

AcceptanceRateMCTuner(
  targetrate::Float64;
  score::Function=logistic_rate_score,
  period::Int=100,
  verbose::Bool=false
) =
  AcceptanceRateMCTuner(targetrate, score, period, verbose)

function tune!(tune::AcceptanceRateMCTune, tuner::AcceptanceRateMCTuner)
  rate!(tune)
  tune.step *= tuner.score(tune.rate-tuner.targetrate)
end
