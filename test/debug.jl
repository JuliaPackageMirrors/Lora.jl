#######################################################################

reload("Lora") ; m = Lora
pwd()

<<<<<<< HEAD
cd( joinpath( Pkg.dir("Lora"), "src/parsers" ) )
include("LoraDSL.jl") ; m = LoraDSL
=======
reload("ReverseDiffSource")
include("../src/parsers/LoraDSL.jl") ; m = LoraDSL
>>>>>>> b0ffd86ec666aa923a886b1530aa7c1c23ef2426

LoraDSL.LLAcc
m.LLAcc

Vector = 4.


#########################################################################
#    testing script for simple examples 
#########################################################################

	using Distributions

	# generate a random dataset
	srand(1)
	n = 1000
	nbeta = 10 
	X = [ones(n) randn((n, nbeta-1))] 
	beta0 = randn((nbeta,))
	Y = rand(n) .< ( 1 ./ (1 .+ exp(X * beta0))) 

	# define model
	ex = quote
		vars ~ Normal(0, 1.0)  
		prob = 1 ./ (1. .+ exp(X * vars)) 
		Y ~ Bernoulli(prob)
	end

#######
<<<<<<< HEAD
	fex = m.parsemodel( ex, vars=zeros(10), debug=true )
	@eval tf(vars) = $fex
	tf(zeros(10))

	fex = m.parsemodel( ex, vars=zeros(10), order=1, debug=true )
	tf = eval(tf)
	tf(zeros(10))

	fex = m.parsemodel( ex, vars=zeros(10), order=2, debug=true )


	vec2var(;init...)

	tf2, dummy = m.parsemodel( ex, vars=zeros(10), order=1 )
	vars = zeros(10)
	tf2(zeros(10))
	typeof(tf2)
	dummy
vars

	v = LoraDSL.LLAcc(0.)
	n = m.ReverseDiffSource.NC

	fex = m.parsemodel( ex, vars=zeros(10) )

	m.ReverseDiffSource.rdiff(fex, vars=zeros(10), order=0, debug=true)


	dex = m.ReverseDiffSource.rdiff( :( sum(similar(vars, Int64)) ), vars=zeros(10), order=1)
	eval( :(vars=zeros(2) ; $dex) )

	tex = :( a= LoraDSL.LLAcc(x) ; a.val )
	dex = m.ReverseDiffSource.rdiff( tex, x=12., order=1, debug=true)
	m.ReverseDiffSource.tocode(dex)
	dex
	eval( :(vars=zeros(2) ; $dex) )

	op = LoraDSL.LLAcc
	op = sin
	isconst(:op)
	isconst(LoraDSL.LLAcc)
	string(op)
	dump(op)
	op.name.name
	typeof(op.name)
	methods(show, (IO, Function,))

		mt = 	try
					mod = fullname(Base.function_module(op))
				catch e
					println(g)
					error("[tocode] cannot spell function $op")
				end

		(mt == (:Base,)) && ( mt = () ) 

		Expr(:call, mexpr( tuple(mt..., symbol(string(op))) ), Any[ valueof(x,n) for x in n.parents[2:end] ]...)


quote  # D:\frtestar\.julia\v0.4\Lora\src\parsers\parsemodel.jl, line 27:
    __acc = LoraDSL.LLAcc(0.0) # line 28:
    begin  # none, line 2:
        __acc += logpdf(Normal(0,1.0),vars) # line 3:
        prob = 1 ./ (1.0 .+ exp(X * vars)) # line 4:
        __acc += logpdf(Bernoulli(prob),Y)
    end # line 29:
    __acc.val
end
=======
	fex = m.parsemodel( ex, vars=zeros(10), debug=true)
	fex = m.parsemodel( ex, vars=zeros(10), debug=true, order=1)
	fex, dummy = m.parsemodel( ex, vars=zeros(10))
	fex, dummy = m.parsemodel( ex, vars=zeros(10), order=1, debug=true )
	fex, dummy = m.parsemodel( ex, vars=zeros(10), order=2, debug=true)

	fex(zeros(10))
>>>>>>> b0ffd86ec666aa923a886b1530aa7c1c23ef2426

	dump( fex )
	dump( fex.args[1].args[2] )
	eval( fex )
	dump( :( LoraDSL.LLAcc(0.) ) )

	llf = eval( fex )
	llf( zeros(10) )

	fex = m.parsemodel( ex, vars=zeros(10) )
	fex( zeros(10) )
	LoraDSL.LLAcc


	g = m.ReverseDiffSource.tograph( :( LoraDSL.LLAcc(0) ) )
	typeof(g.nodes[2].main)
	dump( m.ReverseDiffSource.tocode(g) )

    function ll14981(__beta::Vector{Float64}) # D:\frtestar\.julia\v0.4\Lora\src\parsers\parsemodel.jl, line 38:
        try  # line 39:
            begin 
                _tmp1 = LoraDSL.LLAcc(0.0)
                _tmp1 = _tmp1 + logpdf(Distributions.Normal(0,1.0),vars)
                _tmp1 = _tmp1 + logpdf(Distributions.Bernoulli(1 ./ (1.0 .+ exp(X * vars))),Y)
                (_tmp1.val,)
            end
        catch e # line 41:
            isa(e,LoraDSL.OutOfSupportError) || rethrow(e) # line 42:
            return tuple([-Inf,zeros(0)]...)
        end
    end
	ll14981(zeros(10))
	vars

	eval( :( LoraDSL.LLAcc(0) ) )
	dump( :( LoraDSL.LLAcc(0) ) )

	dump( :( $(fullname(LoraDSL.LLAcc.name.module)).LLAcc(0.) ) )



	m.translate

	mod = m.model(ex, vars=zeros(nbeta), gradient=true)

	mod.eval(zeros(nbeta))
	mod.evalg(zeros(nbeta))

	names(mod)

	m.model(ex, vars=zeros(nbeta), gradient=true, debug=true)
	m.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)

	# different samplers
	res = m.run(m * m.MH(0.05) * m.SerialMC(100:1000))
	res = run(m * HMC(2, 0.1) * SerialMC(100:1000))
	res = run(m * NUTS() * SerialMC(100:1000))
	res = run(m * MALA(0.001) * SerialMC(100:1000))
	# TODO : add other samplers here

	# different syntax
	res = run(m, RWM(), SerialMC(steps=1000, thinning=10, burnin=0))
	res = run(m, HMC(2,0.1), SerialMC(thinning=10, burnin=0))
	res = run(m, HMC(2,0.1), SerialMC(burnin=20))


###############  

### README examples 

	mymodel1 = model(v-> -dot(v,v), init=ones(3))
	mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

	modelxpr = quote
	    v ~ Normal(0, 1)
	end

	mymodel3 = model(modelxpr, v=ones(3))
	mymodel4 = model(modelxpr, gradient=true, v=ones(3))

	mychain  = run(mymodel3, RWM(0.1), m.SerialMC(steps=1000, burnin=100))
	mychain  = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100, thinning=5))
	mychain  = run(mymodel1, RWM(0.1), SerialMC(101:5:1000))
	mychain1 = run(mymodel1 * RWM(0.1) * SerialMC(101:5:1000))

	mychain2 = run(mymodel3, HMC(0.75), SerialMC(1000:10000))

	typeof(run)

	acceptance(mychain2)

	# describe(mychain2)

	ess(mychain2)

	actime(mychain2)

	# var(mychain2)
	# var(mychain2, vtype=:iid)
	# var(mychain2, vtype=:ipse)
	# var(mychain2, vtype=:bm)

	mychain1 = resume(mychain1, steps=10000)

	@test_throws ErrorException run(mymodel3 * MALA(0.1) * SerialMC(1:1000))

	run(mymodel4 * MALA(0.1) * SerialMC(1:1000))

	mychain = run(mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)] * SerialMC(steps=1000)) 
	mychain[2].samples

	mychain = run(mymodel2 * [HMC(i,0.1) for i in 1:5] * SerialMC(steps=1000))

	nmod = 10
	mods = Array(MCMCLikModel, nmod)
	sts = logspace(1, -1, nmod)
	for i in 1:nmod
	  m = quote
	    y = abs(x)
	    y ~ Normal(1, $(sts[i]))
	  end
	  mods[i] = model(m, x=0)
	end

	targets = MCMCTask[mods[i] * RWM(sts[i]) * SeqMC(steps=10, burnin=0) for i in 1:nmod]
	particles = [[randn()] for i in 1:1000]

	mychain3 = run(targets, particles=particles)

	# mychain4 = wsample(mychain3.samples, mychain3.diagnostics["weigths"], 1000)
	# mean(mychain4)



###################### modules ########################################
	a
	module Abcd
		module Abcd2
			type Argf ; end
			function probe()
				println(current_module())
				eval( :( a = 1 ))
				current_module().eval( :( a = 2 ) )
			end
			function probe2()
				println(repr(Argf))
			end
		end
	end

	Abcd.Abcd2.probe()


	Abcd.Abcd2.probe2()
	a

	t = Abcd.Abcd2.Argf
	tn = t.name
	tn.module
	fullname(tn.module)

	t = Abcd.Abcd2.probe2
	string(t)
	t.module

	fullname(t)
	methods(fullname)

	dump( :( Abcd.Abcd2.Argf ))
	zn = symbol("Abcd.Abcd2.Argf")
	dump( :( $zn ))
	tn.names


	mexpr(ns) = length(ns) == 1 ? ns[1] : Expr(:., mexpr(ns[1:end-1]), QuoteNode(ns[end]) )
	dump( mexpr( tuple([fullname(tn.module)..., tn.name ]...) ) )

	dump( mexpr( (:A, :B, :C)) )
	dump( mexpr( (:A,)) )

	ft = sin
	ft.name


	yo = mexpr( tuple([fullname(tn.module)..., tn.name ]...) )

	dump( yo )
	dump( :( $yo ) )

	eval( yo )
	t = eval( :( $yo ) )
	x = t()
	typeof(x)

	repr(t)
	typeof(t.name)
	t.module

	methodswith(TypeName)

	Main.repr( Abcd.Abcd2.Argf )
	Abcd.repr( Abcd.Abcd2.Argf )

	Abcd.Abcd2.repr( Argf )

	help(Base.isvarargtype)

	module Abcd
		x = 3
		probe() = println(x)
	end
	Abcd.probe()

	LoraDSL.LLAcc

	probe() = LoraDSL.LLAcc(0)

	LoraDSL.LLAcc(0)
	probe()

	x

	order=3
	quote
       	return $(order==0 ? -Inf : Expr(:tuple, [-Inf, zeros(order)]...) )
	end
