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

ContinuousUnivariateParameter(1, :p)
ContinuousUnivariateParameter(1, :p, state=ContinuousUnivariateParameterState(Float32))

println("    Testing basic ContinuousMultivariateParameterState constructors...")

ContinuousMultivariateParameterState([1., 1.5])
ContinuousMultivariateParameterState([1., 1.5], MvNormal(ones(2)))
ContinuousMultivariateParameterState(Float32)
ContinuousMultivariateParameterState(Float64)
ContinuousMultivariateParameterState(BigFloat)
ContinuousMultivariateParameterState(Float64, MvNormal(ones(2)))

println("    Testing basic ContinuousMultivariateParameter constructors...")

ContinuousMultivariateParameter(1, :p)
ContinuousMultivariateParameter(1, :p, state=ContinuousMultivariateParameterState(Float32))
