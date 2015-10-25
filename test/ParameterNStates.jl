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

println("    Testing ContinuousUnivariateMCChain constructors and methods...")

nstaten = 4
nstate = ContinuousUnivariateMCChain(Float64, nstaten)

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
for i in 2:13
  @test length(nstate.(functionnames[i])) == 0
end
@test length(nstate.diagnostickeys) == 0
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.n == nstaten

nstaten = 5
nstate = ContinuousUnivariateMCChain(Float32, nstaten, [true; fill(false, 5); true; fill(false, 6)], [:accept])

@test eltype(nstate) == Float32
@test length(nstate.value) == nstaten
@test length(nstate.gradlogtarget) == nstaten
for i in [2:6; 8:13]
  @test length(nstate.(functionnames[i])) == 0
end
@test length(nstate.diagnostickeys) == 1
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.n == nstaten

statev = Float32(3.)
stateglt = Float32(4.21)
state = ContinuousUnivariateParameterState(statev, [:accept], [true])
state.gradlogtarget = stateglt
savei = 2

nstate.copy(state, savei)
@test nstate.value[savei] == statev
@test nstate.gradlogtarget[savei] == stateglt
nstate.diagnosticvalues[savei] == true

nstaten = 10
nstate = ContinuousUnivariateMCChain(Float64, nstaten, [:value, :logtarget], [:accept])

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
@test length(nstate.logtarget) == nstaten
for i in [2:3; 5:13]
  @test length(nstate.(functionnames[i])) == 0
end
@test length(nstate.diagnostickeys) == 1
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.n == nstaten

statev = Float64(1.25)
statelt = Float64(-1.12)
state = ContinuousUnivariateParameterState(statev, [:accept], [false])
state.logtarget = statelt
savei = 7

nstate.copy(state, savei)
@test nstate.value[savei] == statev
@test nstate.logtarget[savei] == statelt
nstate.diagnosticvalues[savei] == false

println("    Testing ContinuousMultivariateMCChain constructors and methods...")

nstatesize = 2
nstaten = 4
nstate = ContinuousMultivariateMCChain(Float64, nstatesize, nstaten)

@test eltype(nstate) == Float64
@test size(nstate.value) == (nstatesize, nstaten)
for i in (2, 3, 4)
  @test length(nstate.(functionnames[i])) == 0
end
for i in 5:7
  @test size(nstate.(functionnames[i])) == (0, 0)
end
for i in 8:10
  @test size(nstate.(functionnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(nstate.(functionnames[i])) == (0, 0, 0, 0)
end
@test length(nstate.diagnostickeys) == 0
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.size == nstatesize
@test nstate.n == nstaten

nstatesize = 2
nstaten = 5
nstate = ContinuousMultivariateMCChain(
  Float32,
  nstatesize,
  nstaten,
  [true; fill(false, 3); true; fill(false, 8)],
  [:accept]
  )

@test eltype(nstate) == Float32
@test size(nstate.value) == (nstatesize, nstaten)
@test size(nstate.gradloglikelihood) == (nstatesize, nstaten)
for i in (2, 3, 4)
  @test length(nstate.(functionnames[i])) == 0
end
for i in (6, 7)
  @test size(nstate.(functionnames[i])) == (0, 0)
end
for i in 8:10
  @test size(nstate.(functionnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(nstate.(functionnames[i])) == (0, 0, 0, 0)
end
@test length(nstate.diagnostickeys) == 1
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.size == nstatesize
@test nstate.n == nstaten

statev = Float32[0.17, 9.44]
stategll = Float32[-0.01, 4.7]
state = ContinuousMultivariateParameterState(statev, [:gradloglikelihood], [:accept], [false])
state.gradloglikelihood = stategll
savei = 3

nstate.copy(state, savei)
@test nstate.value[:, savei] == statev
@test nstate.gradloglikelihood[:, savei] == stategll
nstate.diagnosticvalues[savei] == false

nstatesize = 2
nstaten = 10
nstate = ContinuousMultivariateMCChain(Float16, nstatesize, nstaten, [:value, :logtarget, :gradlogtarget],[:accept])

@test eltype(nstate) == Float16
@test size(nstate.value) == (nstatesize, nstaten)
@test length(nstate.logtarget) == nstaten
@test size(nstate.gradlogtarget) == (nstatesize, nstaten)
for i in (2, 3)
  @test length(nstate.(functionnames[i])) == 0
end
for i in (5, 6)
  @test size(nstate.(functionnames[i])) == (0, 0)
end
for i in 8:10
  @test size(nstate.(functionnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(nstate.(functionnames[i])) == (0, 0, 0, 0)
end
@test length(nstate.diagnostickeys) == 1
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.size == nstatesize
@test nstate.n == nstaten

statev = Float16[6.91, 0.42]
statelt = Float16(4.67)
stateglt = Float16[-0.01, 3.2]
state = ContinuousMultivariateParameterState(statev, [:gradlogtarget], [:accept], [true])
state.logtarget = statelt
state.gradlogtarget = stateglt
savei = 7

nstate.copy(state, savei)
@test nstate.value[:, savei] == statev
@test nstate.logtarget[savei] == statelt
@test nstate.gradlogtarget[:, savei] == stateglt
nstate.diagnosticvalues[savei] == true
