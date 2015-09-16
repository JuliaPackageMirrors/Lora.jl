using Base.Test
using Lora

println("    Testing UnivariateGenericVariableState constructors...")

v = Float64(1.21)
s = UnivariateGenericVariableState(v)

@test eltype(s) == Float64
@test s.value == v

println("    Testing MultivariateGenericVariableState constructors...")

v = Float32[1.5, 4.1]
s = MultivariateGenericVariableState(v)

@test eltype(s) == Float32
@test s.value == v
@test s.size == length(v)

ssize = 3
s = MultivariateGenericVariableState(Float16, ssize)

@test eltype(s) == Float16
@test s.size == ssize

println("    Testing MatrixvariateGenericVariableState constructors...")

v = BigFloat[3.11 7.34; 9.7 6.72; 1.18 8.1]
s = MatrixvariateGenericVariableState(v)

@test eltype(s) == BigFloat
@test s.value == v
@test s.size == size(v)

ssize = (3, 5)
s = MatrixvariateGenericVariableState(Float16, ssize)

@test eltype(s) == Float16
@test s.size == ssize
