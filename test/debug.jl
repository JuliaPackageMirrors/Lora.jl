#######################################################################

reload("ReverseDiffSource")
reload("Lora") ; m = Lora

pwd()

cd( joinpath( Pkg.dir("Lora"), "src/parsers" ) )
include("LoraDSL.jl") ; m = LoraDSL

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
	function generateModelFunction(ex::Expr; gradient=false, debug=false, init...)

		mex, gradient, debug, init = copy(ex), true, false, [(:vars, zeros(nbeta))]

		mex.head != :block && (mex = Expr(:block, mex))  # enclose in block if needed
		length(mex.args)==0 && error("model should have at least 1 statement")

		vsize, pmap, vinit = m.modelVars(;init...) # model param info

		mex2 = m.translate(mex) # rewrite ~ statements
		mex2 = Expr(:block, [ :($(m.ACC_SYM) = LLAcc(0.)), # add log-lik accumulator initialization
			                   mex2.args, 
			                   # :( $ACC_SYM = $(Expr(:., ACC_SYM, Expr(:quote, :val)) ) )]... )
			                   :( $(Expr(:., m.ACC_SYM, Expr(:quote, :val)) ) )]... )

		r = m.ReverseDiffSource
		# g  = r.drules[(logpdf,1)][(AbstractArray{Bernoulli}, AbstractArray)][1]
		# ss = m.ReverseDiffSource.drules[(logpdf,1)][(AbstractArray{Bernoulli}, AbstractArray)][2]

		# g  = m.ReverseDiffSource.drules[(logpdf,2)][(AbstractArray{Bernoulli}, AbstractArray)][1]
		# ss = m.ReverseDiffSource.drules[(logpdf,1)][(AbstractArray{Bernoulli}, AbstractArray)][2]

		LLAcc = m.LLAcc

		g = r.tograph(mex2)
		r.calc!(g, params = Dict(:x    => 1. ) )
		r.calc!(g, params = Dict(:vars => zeros(10)) )
		map( r.zeronode, g.nodes[1:7] )
		r.zeronode( g.nodes[8] )
		g.nodes[6].main.value
		dump( g.nodes[22].main ) 

		v = g.nodes[5].val

		n = g.nodes[22]



			quote 
			    _tmp1 = Lora.LLAcc(0.0)
			    _tmp2 = Distributions.Normal(0,1.0)
			    _tmp3 = X * vars
			    _tmp4 = size(vars)
			    _tmp5 = cell(1)
			    _tmp6 = cell(1000)
			    _tmp7 = cell(1)
			    _tmp8 = cell(1)
			    _tmp9 = similar(Y,Any)
			    _tmp10 = logpdf(_tmp2,vars)
			    _tmp11 = exp(_tmp3)
			    _tmp5[1] = 0.0
			    _tmp7[1] = 0.0
			    _tmp8[1] = 0.0
			    _tmp12 = zeros(_tmp4)
			    _tmp13 = 1.0 .+ _tmp11
			    _tmp14 = size(_tmp10)
			    _tmp1 = _tmp1 + _tmp10
			    _tmp15 = 1 ./ _tmp13
			    _tmp16 = Distributions.Bernoulli(_tmp15)
			    _tmp8[1] = _tmp8[1] + 1.0
			    _tmp17 = logpdf(_tmp16,Y)
			    _tmp18 = _tmp7 + _tmp8
			    _tmp19 = size(_tmp17)
			    _tmp8 = 0.0
			    _tmp1 = _tmp1 + _tmp17
			    for i = 1:length(_tmp16)
			        _tmp20 = cell(1)
			        _tmp20[1] = 0.0
			        _tmp6[i] = _tmp20
			    end
			    _tmp21 = zeros(_tmp19) + fill(_tmp18[1],_tmp19)
			    _tmp22 = _tmp5 + (_tmp8 + _tmp18)
			    for i = 1:length(Y)
			        _tmp9[i] = [1.0 / ((_tmp16[i].p - 1.0) + Y[i])] * _tmp21[i]
			    end
			    _tmp23 = _tmp6 + _tmp9
			    _tmp24 = zeros(_tmp14) + fill(_tmp22[1],_tmp14)
			    for i = 1:length(vars)
			        _tmp12[i] = ((_tmp2.μ - vars[i]) / (_tmp2.σ * _tmp2.σ)) * _tmp24[i]
			    end
			    _tmp25 = zeros(size(_tmp23))
			    for i = 1:length(_tmp23)
			        _tmp25[i] = (_tmp23[i])[1]
			    end
			    (_tmp1.val,((zeros(_tmp4) + _tmp12) + X' * (zeros(size(_tmp3)) + _tmp11 .* (zeros(size(_tmp11)) + (zeros(size(_tmp13)) + -((zeros(size(_tmp15)) + _tmp25)) ./ (_tmp13 .* _tmp13)))),))
			end


		## build function expression
		if gradient  # case with gradient
			# head, body, outsym = ReverseDiffSource.reversediff(model, 
			# 	                                               rv, false, Lora; 
			# 	                                               init...)

			dmodel = m.rdiff(mex2, vars=zeros(10))

			body = [ m.vec2var(;init...),  # assigments beta vector -> model parameter vars
			         dmodel,
			         :(($outsym, $(m.var2vec(;init...))))]

			# enclose in a try block
			body = Expr(:try, Expr(:block, body...),
					          :e, 
					          quote 
					          	if isa(e, OutOfSupportError)
					          		return(-Inf, zero($PARAM_SYM))
					          	else
					          		rethrow(e)
					          	end
					          end)

		else  # case without gradient
			head, body, outsym = ReverseDiffSource.reversediff(model, 
				                                               rv, true, Lora; 
				                                               init...)

			body = [ vec2var(;init...),  # assigments beta vector -> model parameter vars
			         body.args,
			         outsym ]

			# enclose in a try block
			body = Expr(:try, Expr(:block, body...),
					          :e, 
					          quote 
					          	if isa(e, OutOfSupportError)
					          		return(-Inf)
					          	else
					          		rethrow(e)
					          	end
					          end)

		end

		# build and evaluate the let block containing the function and var declarations
		fn = gensym("ll")
		body = Expr(:function, Expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	Expr(:block, body) )
		body = Expr(:let, Expr(:block, :(global $fn), head.args..., body))

		# println("#############\n$body\n############")

		debug ? body : (eval(body) ; (eval(fn), vsize, pmap, vinit) )
	end

### README examples 

	mymodel1 = m.model(v-> -dot(v,v), init=ones(3))
	mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

	modelxpr = quote
	    v ~ Normal(0, 1)
	end

	mymodel3 = model(modelxpr, v=ones(3))
	mymodel4 = model(modelxpr, gradient=true, v=ones(3))

	mychain  = run(mymodel1, m.RWM(0.1), m.SerialMC(steps=1000, burnin=100))
	mychain  = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100, thinning=5))
	mychain  = run(mymodel1, RWM(0.1), SerialMC(101:5:1000))
	mychain1 = run(mymodel1 * RWM(0.1) * SerialMC(101:5:1000))

	mychain2 = run(mymodel2, HMC(0.75), SerialMC(steps=10000, burnin=1000))

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