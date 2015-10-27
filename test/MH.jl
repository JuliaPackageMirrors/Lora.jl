using Base.Test
using Lora

println("    Testing MHState constructors...")

v = Float32(1.5)
pstate = ContinuousUnivariateParameterState(v, [:accept], [true])
sstate = MHState(deepcopy(pstate))

@test eltype(sstate) == ContinuousUnivariateParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 0
@test sstate.tune.proposed == 0
@test isnan(sstate.tune.rate)
@test sstate.count == 1
@test isnan(sstate.ratio)

v = Float64[-6.55, 2.8]
pstate = ContinuousMultivariateParameterState(v)
sstate = MHState(deepcopy(pstate), BasicMCTune(10, 100, 0.1))

@test eltype(sstate) == ContinuousMultivariateParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 10
@test sstate.tune.proposed == 100
@test sstate.tune.rate == 0.1
@test sstate.count == 1
@test isnan(sstate.ratio)

v = Float16[3.16, -2.97, -8.53]
pstate = ContinuousMultivariateParameterState(v)
sstate = MHState(deepcopy(pstate), BasicMCTune(), 150, 0.27)

@test eltype(sstate) == ContinuousMultivariateParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 0
@test sstate.tune.proposed == 0
@test isnan(sstate.tune.rate)
@test sstate.count == 150
@test sstate.ratio == 0.27
