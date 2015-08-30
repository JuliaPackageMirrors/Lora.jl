using Base.Test
using Lora

println("    Testing ContinuousUnivariateMCChain constructors and methods...")

nstaten = 4
nstate = ContinuousUnivariateMCChain(Float64, nstaten)

@test eltype(nstate) == Float64
for i in 1:14
  length(nstate.(Lora.main_state_field_names[i])) == 0
end
@test nstate.n == nstaten

nstaten = 5
nstate = ContinuousUnivariateMCChain(Float32, nstaten, [true, fill(false, 12), true])

@test eltype(nstate) == Float32
@test length(nstate.value) == nstaten
for i in 2:14
  length(nstate.(Lora.main_state_field_names[i])) == 0
end
@test nstate.n == nstaten

statev = float32(3.)
savei = 2

nstate.save(ContinuousUnivariateParameterState(statev, {:accept=>true}), savei)
@test nstate.value[savei] == statev
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
statell = float64(-1.12)
state = ContinuousUnivariateParameterState(statev, {:accept=>true})
state.logtarget = statell
savei = 7

nstate.save(state, savei)
@test nstate.value[savei] == statev
@test nstate.logtarget[savei] == statell
nstate.diagnostics[:accept][savei] == false
