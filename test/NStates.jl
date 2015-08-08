using Base.Test
using Distributions
using Lora

println("    Testing UnivariateGenericVariableNState constructors...")

s = UnivariateGenericVariableNState(Float64[1.25, -4.4, 7.5])
@test eltype(s) == Float64
@test s.value == Float64[1.25, -4.4, 7.5]
@test s.n == 3

save!(s, UnivariateGenericVariableState(float64(5.2)), 2)
@test s.value == Float64[1.25, 5.2, 7.5]

s = convert(UnivariateGenericVariableNState, UnivariateGenericVariableState(float32(1.1)))
@test eltype(s) == Float32
@test s.value == Float32[1.1]
@test s.n == 1
