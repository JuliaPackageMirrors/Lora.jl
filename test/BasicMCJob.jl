using Base.Test
using Lora

# println("    Testing BasicMCJob constructors...")

runner = BasicMCRunner(1000:10000)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(states, j) -> states[j].logtarget = -states[j].value*states[j].value
)
model = single_parameter_likelihood_model(p)

sampler = MH()

tuner = VanillaMCTuner()

job = BasicMCJob(
  runner,
  model,
  sampler,
  tuner,
  1,
  VariableState[ContinuousUnivariateParameterState(1.25, [:accept])],
  Dict{Symbol, Any}(:diagnostics=>[:accept]),
  true,
  false
)
