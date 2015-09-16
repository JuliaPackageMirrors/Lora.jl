using Base.Test
using Lora

println("    Testing ContinuousUnivariateParameterState constructors...")

v = Float64(1.5)
s = ContinuousUnivariateParameterState(v, Dict{Symbol, Bool}(:accept=>true))

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
@test s.size == length(v)

v = Float16[0.24, 5.5, -6.3]
ssize = length(v)
s = ContinuousMultivariateParameterState(v, [:gradlogtarget], Dict{Symbol, Bool}(:accept=>false))

@test eltype(s) == Float16
@test s.value == v
@test isnan(s.logtarget)
@test length(s.gradloglikelihood) == 0
@test length(s.gradlogtarget) == ssize
@test s.size == ssize
@test s.diagnostics[:accept] == false

s = ContinuousMultivariateParameterState(BigFloat)

@test isa(s.value, Vector{BigFloat})
