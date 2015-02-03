#######################################################################


reload("ReverseDiffSource")
reload("Lora") ; m = Lora
# cd( joinpath( Pkg.dir("Lora"), "src/parsers" ) )
# include("LoraDSL.jl") ; m = LoraDSL

#########################################################################
#    testing script for simple examples 
#########################################################################


    # generate a random dataset
    srand(1)
    n = 100
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
	using Distributions

	mod = m.model(ex, vars=zeros(nbeta), order=1)
	dfunc = m.parsemodel(ex, vars=zeros(nbeta), order=1)

    mod.eval( zeros(nbeta) )
    mod.evalg( zeros(nbeta) )

    @time mod = m.model(ex, vars=zeros(nbeta), order=1)
    # n=10   : 0.26s
    # n=100  : 0.27s
    # n=1000 : 0.27s
    # n=10^5 : 0.30s
    # n=10^6 : 0.35s


    mod = m.model(ex, vars=zeros(nbeta), order=2)
    dummy = m.parsemodel(ex, vars=zeros(nbeta), order=2)
    mod.eval( zeros(nbeta) )
    mod.evalg( zeros(nbeta) )  # plante

    @time mod = m.model(ex, vars=zeros(nbeta), order=2)
    # n=10   : 1.9s  / / 1.10s / 1.02s
    # n=100  : 21s  / 17.8s / 9.9s / 8.3s
    # n=1000 : .
    # n=10^5 : .
    # n=10^6 : .

    # n=10   : .
    # n=100  : 8.9s / 8.4s

    Profile.clear()
    @profile mod = m.model(ex, vars=zeros(nbeta), order=2)
    Profile.print()
    Profile.print(format=:flat)
    # dat, dd = Profile.retrieve()
    # dat
    # using StatsBase
    # sc = countmap(dat)
    # worst=sort(collect(sc), by= t -> t[2], rev=true)[1:10]
    # dd[worst[1][1]]

476 ...urce/src/graph.jl; prune!; line: 225

156 ...urce/src/graph.jl; evalsort!; line: 298


1 ...Source/src/graph.jl; prune!; line: 287
1 ...Source/src/graph.jl; prune!; line: 287
4 ...Source/src/graph.jl; prune!; line: 287
1 ...urce/src/graph.jl; prune!; line: 287
2 ...ource/src/graph.jl; prune!; line: 287
738 ...urce/src/graph.jl; prune!; line: 287
2    ...urce/src/graph.jl; prune!; line: 287
66  ...urce/src/graph.jl; prune!; line: 287
1   ...ource/src/graph.jl; prune!; line: 287

4 ...Source/src/graph.jl; prune!; line: 236
1 ...urce/src/graph.jl; prune!; line: 236
1    ...urce/src/graph.jl; prune!; line: 236
700 ...urce/src/graph.jl; prune!; line: 236
1 ...ource/src/graph.jl; prune!; line: 236
1   ...ource/src/graph.jl; prune!; line: 236
102 ...urce/src/graph.jl; prune!; line: 236
1 ...ource/src/graph.jl; prune!; line: 236
700 ...urce/src/graph.jl; prune!; line: 236

#################################################################################
	# mod.eval(zeros(nbeta))
	# mod.evalg(zeros(nbeta))

	job = m.MCJob(:task, mod, m.MH(), m.SerialMC(1:1000))
	c = m.run(job)
	c = m.resume(job, c, 1000)
	m.acceptance(c[1])
	c = m.resume(job, c, 1000, keep=false)



	c = m.run(mod, m.MH(0.1), m.SerialMC(1:1000))
	m.acceptance(c[1])
	c = m.run(mod, m.MH([0.1:0.1:1.0]), m.SerialMC(1:1000))
	m.acceptance(c[1])

	rand(MvNormal(zeros(3), [1., 2., 3]), 4)
	# isa(mj, m.MCJob{:task, m.SerialMCBaseRunner})
	# typeof(mj)
	# isa(mj, m.MCPlainJob)
	# isa(mj, m.MCTaskJob)
	# isa(mj, m.MCTaskJob{m.SerialMC})

	m.run(mj)

	methods(run)

	mj2 = m._MCJob(:plain, mod)
	m.run(mj2)

	s = m.MH()
	s.randproposal
	s.randproposal([4.])
	rand(IsoNormal(mean=[4.], 1.))
	methods(IsoNormal)
 call{Cov<:PDMats.AbstractPDMat,
 	  Mean<:Union(Array{Float64,1},
 	  Distributions.ZeroVector{Float64})}
 	(::Type{ Distributions.MvNormal{Cov<:PDMats.AbstractPDMat,Mean<:Union(Array{Float64,1},Distributions.ZeroVector{Float64})} },
 	 μ,Σ)
 	IsoNormal(zeros(2),eye(2))

 	methods(PDMats.ScalMat)
 	typeof(PDMats.ScalMat)
 	PDMats.ScalMat(2,1.)

 	IsoNormal(zeros(2), PDMats.ScalMat(2,1.))

	methods(m._MCJob, (Symbol, m.MCModel))
	methods(m.run, (m.MCModel))

	res = m.run(mod, m.MH(), m.SerialMC(1:1000))

	res = m.run(m._MCJob(:task, mod))
	, m.MH(), m.SerialMC(1:1000))


	m.MCJob(mod, m.MH(), m.SerialMC(1:1000))

methods(run)	
	res = run(m, HMC(2,0.1), SerialMC(thinning=10, burnin=0))
	res = run(m, HMC(2,0.1), SerialMC(burnin=20))


	broadcast(+, [0, 1], [2 3])
	broadcast(+, [0, 1], [2 3], 4)
	broadcast(vcat, [0, 1], [2 3])

	tuple(1,2)



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



	type Abcd2{T, S}
		x::T
		t::Type(S)
	end

	Abcd2(2.5)
	Abcd2{:test}(2.5)

	Abcd{Float64, :test}(2.5)


abstract PlainJob
abstract TaskJob

### Generic MCJob type is used for Monte Carlo jobs based on coroutines or not 
module Sandbox
	type Typ{T, R}    
	  x::R
	end

	typealias Typa{R}  Typ{:a, R}
	typealias Typb{R}  Typ{:b, R}

	sum(x::Typa) = 0.
	sum(x::Typb) = 1.
end

x = Sandbox.Typa(2)
x = Sandbox.Typa{Float64}(2)

isa(x, Sandbox.Typa)
Sandbox.sum(x)

Abcd3a{T}(x::T) = Abcd3{:a, T}(x)

Abcd3{:a, Real}(2.5)

Abcd3aa(2.5)


sum(t::Abcd3a{Real}) = 2
sum(t::Abcd3a{Int}) = 3

sum(Abcd3a(2.5))

MCPlainJob(m::MCModel, s::MCSampler, r::MCRunner, t::MCTuner) = MCPlainJob([m], [s], [r], [t])
MCTaskJob( m::MCModel, s::MCSampler, r::MCRunner, t::MCTuner) = MCTaskJob( [m], [s], [r], [t])



broadcast( (args...) -> vcat(args...), Any[1,2], Any[3 4])

broadcast_shape()
