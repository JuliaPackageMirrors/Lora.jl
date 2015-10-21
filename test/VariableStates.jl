using Base.Test
using Lora

println("    Testing UnivariateBasicVariableState constructors...")

v = Float64(1.21)
s = UnivariateBasicVariableState(v)

@test eltype(s) == Float64
@test s.value == v

println("    Testing MultivariateBasicVariableState constructors...")

v = Float32[1.5, 4.1]
s = MultivariateBasicVariableState(v)

@test eltype(s) == Float32
@test s.value == v
@test s.size == length(v)

ssize = 3
s = MultivariateBasicVariableState(Float16, ssize)

@test eltype(s) == Float16
@test s.size == ssize

println("    Testing MatrixvariateBasicVariableState constructors...")

v = BigFloat[3.11 7.34; 9.7 6.72; 1.18 8.1]
s = MatrixvariateBasicVariableState(v)

@test eltype(s) == BigFloat
@test s.value == v
@test s.size == size(v)

ssize = (3, 5)
s = MatrixvariateBasicVariableState(Float16, ssize)

@test eltype(s) == Float16
@test s.size == ssize
