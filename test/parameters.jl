using Distributions
using Lora

println("    Testing basic ContinuousUnivariateParameterState constructors...")

ContinuousUnivariateParameterState(1.)
ContinuousUnivariateParameterState(1., Normal())
ContinuousUnivariateParameterState(Float32)
ContinuousUnivariateParameterState(Float64)
ContinuousUnivariateParameterState(BigFloat)
ContinuousUnivariateParameterState(Float64, Beta())

println("    Testing basic ContinuousUnivariateParameter constructors...")

ContinuousUnivariateParameter(
  1,
  :p,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  nothing,
  ContinuousUnivariateParameterState(Float32)
)

ContinuousUnivariateParameter(1, :p)
