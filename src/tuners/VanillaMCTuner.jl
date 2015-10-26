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
