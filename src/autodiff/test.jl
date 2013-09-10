cd("p:/Documents/julia/MCMC.jl/src/autodiff")
cd("~/devl/MCMC.jl/src/autodiff")
pwd()

@windows_only include("p:/Documents/julia/MCMC.jl/src/autodiff/mymod.jl")
@unix_only include("/home/fredo/devl/MCMC.jl/src/autodiff/mymod.jl")

######
	y = 12
	Abcd.generateModelFunction(:( x+y ), gradient=true, x=0, debug=true)
	f, a,b,c = Abcd.generateModelFunction(:( x+y ), gradient=true, x=0)
	f, a,b,c = Abcd.generateModelFunction(:( a=x ), gradient=true, x=0)
	f([-13.])


	y = 1.
	ex = :( x' )

	m = Abcd.ParsingStruct()
	Abcd.setInit!(m, [(:x,[0. 1 ; 1 2])])
	# dump(m)
	Abcd.parseModel!(m, ex)
	# m.source
	Abcd.unfold!(m)
	# m.exprs
	Abcd.uniqueVars!(m)
	Abcd.categorizeVars!(m)

	Abcd.preCalculate(m)
	Abcd.backwardSweep!(m)

	Abcd.vhint
	import Abcd.LLAcc
	for (k,v) in Abcd.vhint
		t = typeof(v)
		println("$k, $t, $(t==LLAcc ? "-" : size(v))")
	end
	m.exprs
	m.dexprs

	f, a,b,c = Abcd.generateModelFunction(:( x+y ), gradient=true, x=[0. 1 ; 1 2])
	f([1.,0,0,0])
	Abcd.generateModelFunction(:( x+y ), gradient=true, x=[0. 1 ; 1 2], debug=true)

	Abcd.eval(:( (symbol("##830"))([1.,2]) ) )

###############

	Abcd.generateModelFunction(ex, gradient=false, debug=true, mu=0.0)
	Abcd.generateModelFunction(ex, gradient=true, debug=true, mu=0.0, sd=1.0)
	Abcd.generateModelFunction(ex, gradient=true, debug=true, mu=0.0)


	f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, mu=0.0, sd=1.0)

	f([0.,1])

	include("mymod.jl")


	tex = quote
		a = sin(nb)
		x = sum(a)
		a[2] = 4+a[3]
		sum(a)
	end

	Abcd.generateModelFunction(tex, gradient=true, debug=true, nb=4.)

	m = Abcd.ParsingStruct()
	Abcd.setInit!(m, [(:nb,4.)])
	Abcd.parseModel!(m, tex)
	Abcd.unfold!(m)
	# m.exprs
	Abcd.uniqueVars!(m)
	Abcd.categorizeVars!(m)
	m.varsset
	m.accanc
	Abcd.preCalculate(m)

	Abcd.backwardSweep!(m)

	m.exprs
	m.dexprs


##### 

	using NumericExtensions

	evaluate{T<:FloatingPoint}(::Myfunc, x::T) = sum(x ./ p - (n-x) ./ (1 - p)) * ds

################
