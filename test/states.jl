using Base.Test
using Distributions
using Lora

println("    Testing generic variable state constructors...")

s = MultivariateGenericVariableState([1.5, 4.1])
@test s.size == 2

s = MatrixvariateGenericVariableState([3.11 7.34; 9.7 6.72; 1.18 8.1])
@test s.size == (3, 2)

println("    Testing ContinuousUnivariateParameterState constructors...")

s = ContinuousUnivariateParameterState(1.5, {:accept=>true})
@test s.value == 1.5
@test isnan(s.logtarget)
@test s.diagnostics[:accept] == true

s = ContinuousUnivariateParameterState(BigFloat)
@test isa(s.value, BigFloat)

println("    Testing ContinuousMultivariateParameterState constructors...")

s = ContinuousMultivariateParameterState([1., 1.5])
@test s.value == [1., 1.5]
@test isnan(s.logtarget)
@test length(s.gradlogtarget) == 0

monitor = Dict{Symbol, Bool}()
monitor[:gradlogtarget] = true
s = ContinuousMultivariateParameterState([1., 1.5], monitor, {:accept=>true})
@test s.value == [1., 1.5]
@test isnan(s.logtarget)
@test length(s.gradloglikelihood) == 0
@test length(s.gradlogtarget) == 2

s = ContinuousMultivariateParameterState(BigFloat)
@test isa(s.value, Vector{BigFloat})
