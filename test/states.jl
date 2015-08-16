using Base.Test
using Distributions
using Lora

println("    Testing generic variable state constructors...")

v = float64(1.21)
s = UnivariateGenericVariableState(v)
@test eltype(s) == Float64
@test s.value == v

v = Float32[1.5, 4.1]
s = MultivariateGenericVariableState(v)
@test eltype(s) == Float32
@test s.value == v
@test s.size == 2

v = BigFloat[3.11 7.34; 9.7 6.72; 1.18 8.1]
s = MatrixvariateGenericVariableState(v)
@test eltype(s) == BigFloat
@test s.value == v
@test s.size == (3, 2)

println("    Testing ContinuousUnivariateParameterState constructors...")

v = float64(1.5)
s = ContinuousUnivariateParameterState(v, {:accept=>true})
@test eltype(s) == Float64
@test s.value == v
@test isnan(s.logtarget)
@test s.diagnostics[:accept] == true

s = ContinuousUnivariateParameterState(Float16)
@test isa(s.value, Float16)

println("    Testing ContinuousMultivariateParameterState constructors...")

v = Float64[1., 1.5]
s = ContinuousMultivariateParameterState(v)
@test eltype(s) == Float64
@test s.value == v
@test isnan(s.logtarget)
@test length(s.gradlogtarget) == 0

monitor = Dict{Symbol, Bool}()
monitor[:gradlogtarget] = true
v = Float32[1., 1.5]
s = ContinuousMultivariateParameterState(v, monitor, {:accept=>false})
@test eltype(s) == Float32
@test s.value == v
@test isnan(s.logtarget)
@test length(s.gradloglikelihood) == 0
@test length(s.gradlogtarget) == 2
@test s.size == 2
@test s.diagnostics[:accept] == false

s = ContinuousMultivariateParameterState(BigFloat)
@test isa(s.value, Vector{BigFloat})
