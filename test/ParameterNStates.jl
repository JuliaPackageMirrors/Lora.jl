using Base.Test
using Lora

println("    Testing ContinuousUnivariateMCChain constructors and methods...")

nstaten = 4
nstate = ContinuousUnivariateMCChain(Float64, nstaten)

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
for i in 2:14
  @test length(nstate.(Lora.main_state_field_names[i])) == 0
end
@test nstate.n == nstaten

nstaten = 5
nstate = ContinuousUnivariateMCChain(Float32, nstaten, [true, fill(false, 5), true, fill(false, 6), true])

@test eltype(nstate) == Float32
@test length(nstate.value) == nstaten
@test length(nstate.gradlogtarget) == nstaten
for i in [2:6, 8:14]
  @test length(nstate.(Lora.main_state_field_names[i])) == 0
end
@test nstate.n == nstaten

statev = float32(3.)
stateglt = float32(4.21)
state = ContinuousUnivariateParameterState(statev, {:accept=>true})
state.gradlogtarget = stateglt
savei = 2

nstate.save(state, savei)
@test nstate.value[savei] == statev
@test nstate.gradlogtarget[savei] == stateglt
nstate.diagnostics[:accept][savei] == true

nstaten = 10
nstate = ContinuousUnivariateMCChain(Float64, nstaten, [:value, :logtarget, :diagnostics])

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
@test length(nstate.logtarget) == nstaten
for i in [2:3, 5:14]
  @test length(nstate.(Lora.main_state_field_names[i])) == 0
end
@test nstate.n == nstaten

statev = float64(1.25)
statelt = float64(-1.12)
state = ContinuousUnivariateParameterState(statev, {:accept=>true})
state.logtarget = statelt
savei = 7

nstate.save(state, savei)
@test nstate.value[savei] == statev
@test nstate.logtarget[savei] == statelt
nstate.diagnostics[:accept][savei] == false

println("    Testing ContinuousMultivariateMCChain constructors and methods...")

nstatesize = 2
nstaten = 4
nstate = ContinuousMultivariateMCChain(Float64, nstatesize, nstaten)

@test eltype(nstate) == Float64
@test size(nstate.value) == (nstatesize, nstaten)
for i in (2, 3, 4, 14)
  @test length(nstate.(Lora.main_state_field_names[i])) == 0
end
for i in 5:7
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0)
end
for i in 8:10
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0, 0, 0)
end
@test nstate.size == nstatesize
@test nstate.n == nstaten

nstatesize = 2
nstaten = 5
nstate = ContinuousMultivariateMCChain(Float32, nstatesize, nstaten, [true, fill(false, 3), true, fill(false, 8), true])

@test eltype(nstate) == Float32
@test size(nstate.value) == (nstatesize, nstaten)
@test size(nstate.gradloglikelihood) == (nstatesize, nstaten)
for i in (2, 3, 4, 14)
  @test length(nstate.(Lora.main_state_field_names[i])) == 0
end
for i in (6, 7)
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0)
end
for i in 8:10
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(nstate.(Lora.main_state_field_names[i])) == (0, 0, 0, 0)
end
@test nstate.size == nstatesize
@test nstate.n == nstaten

statev = Float32[0.17, 9.44]
stategll = Float32[-0.01, 4.7]
state = ContinuousMultivariateParameterState(statev, [:gradloglikelihood], {:accept=>false})
state.gradloglikelihood = stategll
savei = 3

nstate.save(state, savei)
@test nstate.value[:, savei] == statev
@test nstate.gradloglikelihood[:, savei] == stategll
nstate.diagnostics[:accept][savei] == false
