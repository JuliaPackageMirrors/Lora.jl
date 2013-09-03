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
# m.source
Abcd.unfold!(m)
# m.exprs
Abcd.uniqueVars!(m)
Abcd.categorizeVars!(m)

Abcd.preCalculate(m)

Abcd.backwardSweep!(m)

m.exprs
m.dexprs

x = Abcd.hint(symbol("##tmp#254") )

dump(:(x.mean))
dump(:(Abcd.X))


Abcd.d_logpdf_x1

Abcd.d_logpdf_x1(x, 12)

Abcd.derive(:(logpdf($(symbol("##tmp#254")), 1.)), 1, :acc)

opex = :(logpdf($(symbol("##tmp#364")), X))
index = 1
dsym = :acc


	const ACC_SYM = :__acc       # name of accumulator variable
	const PARAM_SYM = :__beta    # name of parameter vector
	const TEMP_NAME = "tmp"      # prefix of temporary variables in log-likelihood function
	const DERIV_PREFIX = "d"     # prefix of gradient variables in log-likelihood function

	vs = opex.args[1+index]
	ds = symbol("$DERIV_PREFIX$dsym")
	args = opex.args[2:end]
	
	val = map(Abcd.hint, args) ; # get sample values of args to find correct gradient statement

	fn = symbol("d_$(opex.args[1])_x$index")

	# try
		dexp = Abcd.eval(Expr(:call, fn, val...))

		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		dexp = Abcd.substSymbols(dexp, smap)

		# unfold for easier optimization later
	    m = Abcd.ParsingStruct()
	    m.source = :(dummy = $dexp )

Abcd.toExprH(:(sin(x)))
Abcd.toExprH(:(x.mmean))



		Abcd.unfold!(m)  
		m.exprs[end] = m.exprs[end].args[2] # remove last assignment

		m.exprs[end] = :( $(symbol("$DERIV_PREFIX$vs")) = $(symbol("$DERIV_PREFIX$vs")) + $(m.exprs[end]) )
		return m.exprs
	catch e 
		error("[derive] Doesn't know how to derive $opex by argument $vs")
	end



###############

Abcd.generateModelFunction(ex, gradient=false, debug=true, mu=0.0)
Abcd.generateModelFunction(ex, gradient=true, debug=true, mu=0.0)

f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, mu=0.0, sd=1.0)

f([0.,1])


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


##### idée : type spécifique pour l'accu  #####

immutable LLacc
	x::Float64
	function LLacc(x::Real)
		isfinite(x) || error("give up eval")
		new(x)
	end
end
+(ll::LLacc, x::Real) = LLacc(ll.x + x)
+(ll::LLacc, x::Array{Float64}) = LLacc(ll.x + sum(x))

function f1()
	acc = 0.0
	acc += -10.
	acc += sum(randn(1000))
	acc
end

function f2()
	acc = LLacc(0.)
	acc += -10.
	acc += randn(1000)
	acc.x
end

f1()
f2()

@time for i in 1:100_000; f1(); end  # 4.6 sec
@time for i in 1:100_000; f2(); end  # 4.7 sec, almost no penalty !!!

##### 

using NumericExtensions

evaluate{T<:FloatingPoint}(::Myfunc, x::T) = sum(x ./ p - (n-x) ./ (1 - p)) * ds



