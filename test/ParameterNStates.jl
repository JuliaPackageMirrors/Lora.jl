using Base.Test
using Lora

nstate = ContinuousUnivariateParameterNState{Float64}(Float64, 4)
state = ContinuousUnivariateParameterState(3.)
nstate.save(state, 2)

# TODO:
# done 1) Move fields::Tuple{Symbol} out of functions to avoid redefining it
# 2) Define constructors of the form NState(n::Int) in NSTates.jl
# 3) Fix the {Float64} parameterization issue in ContinuousUnivariateParameterNState constructor
# 4) Define one more constructor for ContinuousUnivariateParameterNState using value input argument
# 5) Complete the tests in the current file for current methods for ContinuousUnivariateParameterNState
# 6) typealias MCChain
