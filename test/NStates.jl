using Base.Test
using Distributions
using Lora

println("    Testing UnivariateGenericVariableNState constructors and methods...")

v = Float64[1.25, -4.4, 7.5]
s = UnivariateGenericVariableNState(v)
@test eltype(s) == Float64
@test s.value == v
@test s.n == 3

save!(s, UnivariateGenericVariableState(float64(5.2)), 2)
@test s.value == Float64[1.25, 5.2, 7.5]

v = float32(1.1)
s = convert(UnivariateGenericVariableNState, UnivariateGenericVariableState(v))
@test eltype(s) == Float32
@test s.value == Float32[v]
@test s.n == 1

println("    Testing MultivariateGenericVariableNState constructors and methods...")

v= Float64[1.35 3.7 4.5; 5.6 8.81 9.2]
s = MultivariateGenericVariableNState(v)
@test eltype(s) == Float64
@test s.value == v
@test s.size == 2
@test s.n == 3

save!(s, MultivariateGenericVariableState(Float64[5.2, 3.31]), 2)
@test s.value == Float64[1.35 5.2 4.5; 5.6 3.31 9.2]

v = Float16[-0.2, -1.12]
s = convert(MultivariateGenericVariableNState, MultivariateGenericVariableState(v))
@test eltype(s) == Float16
@test s.value == reshape(v, 2, 1)
@test s.size == 2
@test s.n == 1
