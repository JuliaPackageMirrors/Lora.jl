using Base.Test
using Lora

println("    Testing MHState constructors...")

v = Float64(1.5)
pstate = ContinuousUnivariateParameterState(v, [:accept], [true])
sstate = MHState(deepcopy(pstate))

@test isequal(sstate.pstate, pstate)
# @test sstate.tune == BasicMCTune()
@test sstate.count == 0
@test isnan(sstate.ratio)
