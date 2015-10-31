using Base.Test
using Lora

# println("    Testing BasicMCJob constructors...")

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(state, states) -> state.logtarget = -state.value*state.value
)
model = single_parameter_likelihood_model(p)

sampler = MH()

tuner = VanillaMCTuner()

mcrange = BasicMCRange(1000:10000)

job = BasicMCJob(
  model,
  1,
  sampler,
  tuner,
  mcrange,
  VariableState[ContinuousUnivariateParameterState(1.25, [:accept])],
  Dict{Symbol, Any}(:diagnostics=>[:accept]),
  true,
  false
)
