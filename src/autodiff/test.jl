cd("p:/Documents/julia/MCMC.jl/src/autodiff")
pwd()

include("mymod.jl")

ex = quote
	X ~ Normal(mu, sd)
end ;

m = Abcd.ParsingStruct()
Abcd.setInit!(m, [(:mu,0.), (:sd, 1.)])
dump(m)
Abcd.parseModel!(m, ex)
m.source
Abcd.unfold!(m)
m.exprs
Abcd.uniqueVars!(m)
Abcd.categorizeVars!(m)

Abcd.preCalculate(m)

Abcd.generateModelFunction(ex, gradient=false, debug=true, mu=0.0)
Abcd.generateModelFunction(ex, gradient=true, debug=true, mu=0.0)

dump(m)

dump(m.source,11)

using Distributions
__acc = 0.0
mu=0. ; sd = 1.

X = randn(1000) ;

eval(m.source)

Abcd.setInit!(m, init)

m
ParsingStruct

## checks initial values
setInit!(m, init)

## rewrites ~ , do some formatting ... on the model expression
parseModel!(m, model)


Abcd.whos()

