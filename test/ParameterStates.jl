using Base.Test
using Lora

functionnames = (
  :value,
  :loglikelihood,
  :logprior,
  :logtarget,
  :gradloglikelihood,
  :gradlogprior,
  :gradlogtarget,
  :tensorloglikelihood,
  :tensorlogprior,
  :tensorlogtarget,
  :dtensorloglikelihood,
  :dtensorlogprior,
  :dtensorlogtarget
)

println("    Testing ContinuousUnivariateParameterState constructors and methods...")

v = Float64(1.5)
s = ContinuousUnivariateParameterState(v, [:accept], [true])

@test eltype(s) == Float64
@test s.value == v
for i in 2:13
  @test isnan(s.(functionnames[i]))
end
@test s.diagnostickeys == [:accept]
@test s.diagnosticvalues == [true]

s = ContinuousUnivariateParameterState(Symbol[], Float16)

@test isa(s.value, Float16)
for i in 1:13
  @test isnan(s.(functionnames[i]))
end
@test s.diagnostickeys == Symbol[]
@test s.diagnosticvalues == []

z = ContinuousUnivariateParameterState(Float64(-3.1), [:accept], [false])
w = deepcopy(z)
@test isequal(z, w)

println("    Testing ContinuousMultivariateParameterState constructors and methods...")

v = Float64[1., 1.5]
s = ContinuousMultivariateParameterState(v)

@test eltype(s) == Float64
@test s.value == v
for i in 2:4
  @test isnan(s.(functionnames[i]))
end
for i in 5:7
  @test length(s.(functionnames[i])) == 0
end
for i in 8:10
  @test size(s.(functionnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(functionnames[i])) == (0, 0, 0)
end
@test s.size == length(v)

v = Float16[0.24, 5.5, -6.3]
ssize = length(v)
s = ContinuousMultivariateParameterState(v, [:gradlogtarget], [:accept], [false])

@test eltype(s) == Float16
@test s.value == v
@test length(s.gradlogtarget) == ssize
for i in 2:4
  @test isnan(s.(functionnames[i]))
end
for i in 5:6
  @test length(s.(functionnames[i])) == 0
end
for i in 8:10
  @test size(s.(functionnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(functionnames[i])) == (0, 0, 0)
end
@test s.size == ssize
@test s.diagnostickeys == [:accept]
@test s.diagnosticvalues == [false]

ssize = 4
s = ContinuousMultivariateParameterState(ssize, Symbol[], Symbol[], BigFloat)

@test isa(s.value, Vector{BigFloat})
@test length(s.value) == ssize
for i in 2:4
  @test isnan(s.(functionnames[i]))
end
for i in 5:7
  @test length(s.(functionnames[i])) == 0
end
for i in 8:10
  @test size(s.(functionnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(functionnames[i])) == (0, 0, 0)
end
@test s.diagnostickeys == Symbol[]
@test s.diagnosticvalues == []

z = ContinuousMultivariateParameterState(Float64[-0.12, 12.15])
w = deepcopy(z)
@test isequal(z, w)
