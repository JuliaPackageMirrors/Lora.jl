using Base.Test
using Lora

# println("    Testing BasicMCJob constructors...")

p = ContinuousMultivariateParameter(
  1,
  :p,
  logtarget=(state, states) -> state.logtarget = dot(state.value, state.value)
)
model = single_parameter_likelihood_model(p)

sampler = MH([1., 1.])

tuner = VanillaMCTuner()

mcrange = BasicMCRange(1000:10000)

vstate = VariableState[ContinuousMultivariateParameterState([1.25, 3.11], [:accept])]

diagnostickeys = Dict{Symbol, Any}(:diagnostics=>[:accept])

job = BasicMCJob(
  model,
  1,
  sampler,
  tuner,
  mcrange,
  vstate,
  diagnostickeys,
  true,
  false
)
