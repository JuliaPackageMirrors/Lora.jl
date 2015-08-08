using Base.Test
using Distributions
using Lora

println("    Testing generic variable state constructors...")

s = UnivariateGenericVariableState(float64(1.21))
@test eltype(s) == Float64

s = MultivariateGenericVariableState(Float32[1.5, 4.1])
@test eltype(s) == Float32
@test s.size == 2

s = MatrixvariateGenericVariableState(BigFloat[3.11 7.34; 9.7 6.72; 1.18 8.1])
@test eltype(s) == BigFloat
@test s.size == (3, 2)

println("    Testing ContinuousUnivariateParameterState constructors...")

s = ContinuousUnivariateParameterState(float64(1.5), {:accept=>true})
@test eltype(s) == Float64
@test s.value == 1.5
@test isnan(s.logtarget)
@test s.diagnostics[:accept] == true

s = ContinuousUnivariateParameterState(Float16)
@test isa(s.value, Float16)

println("    Testing ContinuousMultivariateParameterState constructors...")

s = ContinuousMultivariateParameterState(Float64[1., 1.5])
@test eltype(s) == Float64
@test s.value == [1., 1.5]
@test isnan(s.logtarget)
@test length(s.gradlogtarget) == 0

monitor = Dict{Symbol, Bool}()
monitor[:gradlogtarget] = true
s = ContinuousMultivariateParameterState(Float32[1., 1.5], monitor, {:accept=>false})
@test eltype(s) == Float32
@test s.value == [1., 1.5]
@test isnan(s.logtarget)
@test length(s.gradloglikelihood) == 0
@test length(s.gradlogtarget) == 2
@test s.size == 2
@test s.diagnostics[:accept] == false

s = ContinuousMultivariateParameterState(BigFloat)
@test isa(s.value, Vector{BigFloat})
