using Base.Test
using Lora

# nstate = ContinuousUnivariateMCChain(Float64, 4)
# nstate = ContinuousUnivariateMCChain(Float64, 4, [true, fill(false, 12), true])
monitor = Dict{Symbol, Bool}()
monitor[:value] = true
monitor[:diagnostics] = true
nstate = ContinuousUnivariateMCChain(Float64, 4, monitor)
state = ContinuousUnivariateParameterState(3., {:accept=>true})
nstate.save(state, 2)
