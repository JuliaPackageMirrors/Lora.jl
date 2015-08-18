using Base.Test
using Lora

# nstate = ContinuousUnivariateParameterNState(Float64, 4)
nstate = ContinuousUnivariateParameterNState(Float64, 4, [true, fill(false, 12), true])
state = ContinuousUnivariateParameterState(3., {:accept=>true})
nstate.save(state, 2)

# TODO:
# done 1) Move fields::Tuple{Symbol} out of functions to avoid redefining it
# 2) Define constructors of the form NState(n::Int) in NSTates.jl
# done 3) Fix the {Float64} parameterization issue in ContinuousUnivariateParameterNState constructor
# 4) Define one more constructor for ContinuousUnivariateParameterNState using value input argument
# 5) Complete the tests in the current file for current methods for ContinuousUnivariateParameterNState
# 6) typealias MCChain
# skip for now 7) Add monitor field to ParameterNStates 
# done using code generation 8) Copy over diagnostics::Dict entries from state to nstate
