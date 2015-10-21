# # Placeholder for defining BasicMCJob
# # Alternative name for BasicMCJob: SingleParameterLikelihoodSerialMCJob
#
# type BasicMCJob <: MCJob
#   model::GenericModel
#   sampler::MCSampler
#   runner::MCRunner
#   tuner::MCTuner
#   stash::MCSamplerState # MCSamplerState represents the samplers' interlal state
#                         # Alternative names: MCSamplerInternalState, MCSamplerStash
#   send::Function
#   receive::Function
#   reset::Function
#   plain::Bool # If true then don't coroutines, otherwise use tasks for controlling flow of Monte Carlo simulation
#   imperative::Bool # If true then traverse graph imperatively, otherwise declaratively via topological sorting
#   task::Union{Task, Void}
#
#   function BasicMCJob(m::GenericModel s::MCSampler r::MCRunner t::MCTuner)
#   end
# end
#
# # typealias BasicMCJob SingleParameterLikelihoodSerialMCJob
