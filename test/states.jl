using Base.Test
using Distributions
using Lora

println("    Testing generic variable state constructors...")

s = MultivariateGenericVariableState([1.5, 4.1])
@test s.size == 2

s = MatrixvariateGenericVariableState([3.11 7.34; 9.7 6.72; 1.18 8.1])
@test s.size == (3, 2)

println("    Testing ContinuousUnivariateParameterState constructors...")

ContinuousUnivariateParameterState(1.)
ContinuousUnivariateParameterState(Float32)
ContinuousUnivariateParameterState(Float64)
ContinuousUnivariateParameterState(BigFloat)

println("    Testing ContinuousMultivariateParameterState constructors...")

ContinuousMultivariateParameterState([1., 1.5])
ContinuousMultivariateParameterState(Float32)
ContinuousMultivariateParameterState(Float64)
ContinuousMultivariateParameterState(BigFloat)
