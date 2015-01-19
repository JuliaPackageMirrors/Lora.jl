#######################################################################

reload("ReverseDiffSource")
reload("Lora") ; m = Lora

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

		x = 0.5
		mex2 = 	quote 
					__acc = LLAcc(0.0)
					__acc += logpdf(Bernoulli(x), [1, 0])
					__acc.val
				end
		m.rdiff(mex2 , x=0.23)


		quote  # x = 0.3
		    _tmp1 = 0.0
		    _tmp2 = Lora.LLAcc(0.0)
		    _tmp3 = Distributions.Bernoulli(x)
		    _tmp4 = [1,0]
		    _tmp5 = cell(1)
		    _tmp6 = cell(1)
		    _tmp7 = cell(1)
		    _tmp8 = logpdf(_tmp3,_tmp4)
		    _tmp5[1] = 0.0
		    _tmp6[1] = 0.0
		    _tmp7[1] = 0.0
		    _tmp9 = size(_tmp8)
		    _tmp2 = _tmp2 + _tmp8
		    _tmp7[1] = _tmp7[1] + 1.0
		    _tmp10 = _tmp6 + _tmp7
		    _tmp11 = zeros(_tmp9) + fill(_tmp10[1],_tmp9)
		    for i = 1:length(_tmp4)  # i = 1
		        _tmp1 = _tmp1 + [1.0 / ((_tmp3.p - 1.0) + _tmp4[i])] * _tmp11[i]
		    end
		    _tmp12 = _tmp5 + _tmp1
		    (_tmp2.val,(_tmp12[1],))
		end


		g = r.tograph(mex2)
		r.calc!(g, params = Dict(:x    => 1. ) )
		r.calc!(g, params = Dict(:vars => zeros(10)) )
		map( r.zeronode, g.nodes[1:7] )
		r.zeronode( g.nodes[8] )
		g.nodes[6].main.value
		dump( g.nodes[22].main ) 

		v = g.nodes[5].val

		n = g.nodes[22]

		dmodel = m.rdiff(mex2, vars=zeros(10))


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
			head, body, outsym = ReverseDiffSource.reversediff(model, 
				                                               rv, false, Lora; 
				                                               init...)

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

	mymodel1 = model(v-> -dot(v,v), init=ones(3))
	mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

	modelxpr = quote
	    v ~ Normal(0, 1)
	end

	mymodel3 = model(modelxpr, v=ones(3))
	mymodel4 = model(modelxpr, gradient=true, v=ones(3))

	mychain = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100))
	mychain = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100, thinning=5))
	mychain = run(mymodel1, RWM(0.1), SerialMC(101:5:1000))
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
