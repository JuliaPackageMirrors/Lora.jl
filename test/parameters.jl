using Distributions
using Lora

println("    Testing basic ContinuousUnivariateParameterState constructors...")

ContinuousUnivariateParameterState(1.)
ContinuousUnivariateParameterState(Float32)
ContinuousUnivariateParameterState(Float64)
ContinuousUnivariateParameterState(BigFloat)

println("    Testing basic ContinuousUnivariateParameter constructors...")

ContinuousUnivariateParameter(1, :p)
ContinuousUnivariateParameter(1, :p, pdf=Normal())


println("    Testing basic ContinuousMultivariateParameterState constructors...")

ContinuousMultivariateParameterState([1., 1.5])
ContinuousMultivariateParameterState(Float32)
ContinuousMultivariateParameterState(Float64)
ContinuousMultivariateParameterState(BigFloat)

println("    Testing basic ContinuousMultivariateParameter constructors...")

ContinuousMultivariateParameter(1, :p)
ContinuousMultivariateParameter(1, :p, pdf=MvNormal(ones(2)))
