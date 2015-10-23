### Auxiliary functions used as scores for penalising deviation of observed from target acceptance rate

sigmoid(x::Number, a::Number=7.) = 2/(1+exp(-a*x))

### AcceptanceRateMCTune holds the tuning-related local variables of a MCSampler that uses the AcceptanceRateMCTune

type AcceptanceRateMCTune <: MCTune
  step::Float64 # Stepsize of Monte Carlo iteration (for ex leapfrog or drift stepsize)
  accepted::Int # Number of accepted Monte Carlo samples during current tuning period
  proposed::Int # Number of proposed Monte Carlo samples during current tuning period
  rate::Float64 # Observed acceptance rate over tuning period

  function AcceptanceRateMCTune(step::Float64, accepted::Int, proposed::Int, rate::Float64)
    @assert step > 0 "Stepsize of Monte Carlo iteration should be positive"
    @assert accepted >= 0 "Number of accepted Monte Carlo samples should be non-negative"
    @assert proposed >= 0 "Number of proposed Monte Carlo samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(step, accepted, proposed, rate)
  end
end

AcceptanceRateMCTune(step::Float64, accepted::Int=0, proposed::Int=0) =
  AcceptanceRateMCTune(step, accepted, proposed, NaN)

reset!(tune::AcceptanceRateMCTune) = ((tune.accepted, tune.proposed) = (0, 0))

count!(tune::AcceptanceRateMCTune) = (tune.accepted += 1)

rate!(tune::AcceptanceRateMCTune) = (tune.rate = tune.accepted/tune.proposed)

### AcceptanceRateMCTuner tunes empirically on the basis of the discrepancy between observed and target acceptance rate
### This discrepancy is pernalised via an optional score function
### The default score function is a stretched sigmoid

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
  score::Function=sigmoid,
  period::Int=100,
  verbose::Bool=false
) =
  AcceptanceRateMCTuner(targetrate, score, period, verbose)

function tune!(tune::AcceptanceRateMCTune, tuner::AcceptanceRateMCTuner)
  rate!(tune)
  tune.step *= tuner.score(tune.rate-tuner.targetrate)
end
