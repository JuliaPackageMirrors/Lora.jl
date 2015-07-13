using Distributions
using Lora

println("    Testing ContinuousUnivariateParameter constructors...")

ContinuousUnivariateParameter(1, :p)
ContinuousUnivariateParameter(1, :p, pdf=Normal())

println("    Testing ContinuousMultivariateParameter constructors...")

ContinuousMultivariateParameter(1, :p)
ContinuousMultivariateParameter(1, :p, pdf=MvNormal(ones(2)))
