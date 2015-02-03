#######################################################################


reload("ReverseDiffSource")
reload("Lora") ; m = Lora
cd( joinpath( Pkg.dir("Lora"), "src/parsers" ) )
include("LoraDSL.jl") ; m = LoraDSL

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

    Profile.clear()
    @profile mod = m.model(ex, vars=zeros(nbeta), order=2)
    dat, dd = Profile.retrieve()
    dat
    using StatsBase
    sc = countmap(dat)
    worst=sort(collect(sc), by= t -> t[2], rev=true)[1:10]

    dd[worst[1][1]]

    a = sin
    b = sin
    is(a,b)
    a,b = 1., 1
    is(a,b) # false
    a,b = 4+5im, 4+5im
    is(a,b) # true

    Profile.print()

    function fmod(__beta::Vector{Float64}) 
        try  # line 38:
            begin 
                _tmp1 = 1:10
                _tmp2 = Distributions.Normal(0,1.0)
                _tmp3 = Lora.LoraDSL.LLAcc(0.0)
                _tmp4 = cell(1)
                _tmp5 = cell(1)
                _tmp6 = cell(1)
                _tmp7 = similar(Y,Any)
                _tmp8 = X'
                _tmp9 = length(__beta)
                _tmp10 = zeros(size(__beta))
                _tmp4[1] = 0.0
                _tmp5[1] = 0.0
                _tmp6[1] = 0.0
                _tmp11 = 1:length(Y)
                _tmp12 = Distributions.logpdf(_tmp2,__beta[_tmp1])
                _tmp13 = X * __beta[_tmp1]
                _tmp14 = size(__beta[_tmp1])
                _tmp15 = zeros((_tmp9,_tmp9))
                _tmp16 = exp(_tmp13)
                _tmp17 = size(_tmp12)
                _tmp18 = zeros(_tmp14)
                _tmp19 = 1:length(__beta[_tmp1])
                _tmp3 = _tmp3 + _tmp12
                _tmp20 = 1.0 .+ _tmp16
                _tmp6[1] = _tmp6[1] + 1.0
                _tmp21 = 1 ./ _tmp20
                _tmp22 = _tmp5 + _tmp6
                _tmp23 = _tmp20 .* _tmp20
                _tmp24 = Distributions.Bernoulli(_tmp21)
                _tmp6 = 0.0
                _tmp25 = Distributions.logpdf(_tmp24,Y)
                _tmp26 = cell(size(_tmp24))
                _tmp27 = size(_tmp25)
                _tmp28 = _tmp4 + (_tmp6 + _tmp22)
                _tmp3 = _tmp3 + _tmp25
                _tmp29 = zeros(_tmp27) + Base.Graphics.fill(_tmp22[1],_tmp27)
                for i = 1:length(_tmp26)
                    _tmp30 = cell(1)
                    _tmp30[1] = 0.0
                    _tmp26[i] = _tmp30
                end
                for i = _tmp11
                    _tmp7[i] = [1.0 / ((_tmp24[i].p - 1.0) + Y[i])] * _tmp29[i]
                end
                _tmp31 = zeros(_tmp17) + Base.Graphics.fill(_tmp28[1],_tmp17)
                _tmp32 = _tmp7
                for i = _tmp19
                    _tmp18[i] = ((_tmp2.μ - (__beta[_tmp1])[i]) / (_tmp2.σ * _tmp2.σ)) * _tmp31[i]
                end
                _tmp33 = _tmp26 + _tmp32
                _tmp34 = _tmp18
                _tmp35 = zeros(size(_tmp33))
                _tmp36 = 1:length(_tmp33)
                for i = _tmp36
                    _tmp35[i] = (_tmp33[i])[1]
                end
                _tmp37 = _tmp35
                _tmp38 = zeros(size(_tmp21)) + _tmp37
                _tmp39 = -_tmp38
                _tmp40 = _tmp39 ./ _tmp23
                _tmp41 = zeros(size(_tmp20)) + _tmp40
                _tmp42 = zeros(size(_tmp16)) + _tmp41
                _tmp43 = _tmp16 .* _tmp42
                _tmp44 = zeros(size(_tmp13)) + _tmp43
                _tmp45 = _tmp8 * _tmp44
                _tmp46 = zeros(_tmp14) + _tmp45
                _tmp47 = _tmp46 + _tmp34
                _tmp48 = _tmp10[_tmp1] + _tmp47
                _tmp10[_tmp1] = _tmp48
                for _idx1 = 1:_tmp9
                    _tmp49 = size(__beta[_tmp1])
                    _tmp50 = size(_tmp12)
                    _tmp51 = cell(1)
                    _tmp52 = size(_tmp25)
                    _tmp53 = cell(1)
                    _tmp54 = cell(1)
                    _tmp55 = cell(10)
                    _tmp56 = cell(10)
                    _tmp57 = cell(1)
                    _tmp58 = cell(1)
                    _tmp59 = cell(1)
                    _tmp60 = cell(1)
                    _tmp61 = cell(1)
                    _tmp62 = cell(1)
                    _tmp63 = cell(1)
                    _tmp64 = cell(1)
                    _tmp65 = cell(1)
                    _tmp66 = cell(1)
                    _tmp67 = similar(Y,Any)
                    _tmp68 = zeros(size(__beta))
                    _tmp69 = zeros(_tmp49)
                    _tmp51[1] = 0.0
                    _tmp70 = cell(size(_tmp24))
                    _tmp53[1] = 0.0
                    _tmp54[1] = 0.0
                    _tmp57[1] = 0.0
                    _tmp58[1] = 0.0
                    _tmp59[1] = 0.0
                    _tmp60[1] = 0.0
                    _tmp61[1] = 0.0
                    _tmp62[1] = 0.0
                    _tmp63[1] = 0.0
                    _tmp64[1] = 0.0
                    _tmp65[1] = 0.0
                    _tmp66[1] = 0.0
                    _tmp71 = zeros(size(_tmp10))
                    _tmp72 = zeros(_tmp49)
                    _tmp56[1] = _tmp57
                    _tmp73 = _tmp53 + _tmp54
                    _tmp55[1] = zeros(size(_tmp32[1]))
                    _tmp56[2] = _tmp58
                    _tmp54 = 0.0
                    for i = 1:length(_tmp70)
                        _tmp74 = cell(1)
                        _tmp74[1] = 0.0
                        _tmp70[i] = _tmp74
                    end
                    _tmp55[2] = zeros(size(_tmp32[2]))
                    _tmp56[3] = _tmp59
                    _tmp71[_idx1] = _tmp71[_idx1] + 1.0
                    _tmp75 = _tmp70
                    _tmp55[3] = zeros(size(_tmp32[3]))
                    _tmp56[4] = _tmp60
                    _tmp76 = zeros(_tmp52) + Base.Graphics.fill(_tmp73[1],_tmp52)
                    _tmp77 = _tmp51 + (_tmp54 + _tmp73)
                    _tmp55[4] = zeros(size(_tmp32[4]))
                    _tmp56[5] = _tmp61
                    for i = 1:length(Y)
                        _tmp67[i] = [1.0 / ((_tmp24[i].p - 1.0) + Y[i])] * _tmp76[i]
                    end
                    _tmp55[5] = zeros(size(_tmp32[5]))
                    _tmp56[6] = _tmp62
                    _tmp78 = zeros(size(_tmp47)) + (zeros(size(_tmp48)) + _tmp71[_tmp1])
                    _tmp55[6] = zeros(size(_tmp32[6]))
                    _tmp56[7] = _tmp63
                    _tmp79 = zeros(size(_tmp34)) + _tmp78
                    _tmp80 = zeros(_tmp50) + Base.Graphics.fill(_tmp77[1],_tmp50)
                    _tmp55[7] = zeros(size(_tmp32[7]))
                    _tmp56[8] = _tmp64
                    for i = reverse(_tmp19)
                        _tmp81 = _tmp79[i]
                        _tmp79[i] = 0.0
                        _tmp69[i] = _tmp69[i] + -((_tmp31[i] * _tmp81) / (_tmp2.σ * _tmp2.σ))
                    end
                    for i = 1:length(__beta[_tmp1])
                        _tmp72[i] = ((_tmp2.μ - (__beta[_tmp1])[i]) / (_tmp2.σ * _tmp2.σ)) * _tmp80[i]
                    end
                    _tmp55[8] = zeros(size(_tmp32[8]))
                    _tmp56[9] = _tmp65
                    _tmp55[9] = zeros(size(_tmp32[9]))
                    _tmp56[10] = _tmp66
                    _tmp55[10] = zeros(size(_tmp32[10]))
                    _tmp82 = zeros(size(_tmp43)) + (zeros(size(_tmp44)) + _tmp8' * (zeros(size(_tmp45)) + (zeros(size(_tmp46)) + _tmp78)))
                    _tmp83 = zeros(size(_tmp40)) + (zeros(size(_tmp41)) + (zeros(size(_tmp42)) + _tmp16 .* _tmp82))
                    _tmp84 = _tmp20 .* (zeros(size(_tmp23)) + (-_tmp39 .* _tmp83) ./ (_tmp23 .* _tmp23))
                    _tmp85 = zeros(size(_tmp37)) + (zeros(size(_tmp38)) + -((zeros(size(_tmp39)) + _tmp83 ./ _tmp23)))
                    for i = reverse(_tmp36)
                        _tmp86 = cell(1)
                        _tmp87 = _tmp85[i]
                        _tmp86[1] = 0.0
                        _tmp85[i] = 0.0
                        _tmp86[1] = _tmp86[1] + _tmp87
                        _tmp56[i] = _tmp56[i] + _tmp86
                    end
                    _tmp88 = _tmp55 + _tmp56
                    for i = reverse(_tmp11)
                        _tmp89 = cell(1)
                        _tmp90 = _tmp88[i]
                        _tmp89[1] = 0.0
                        _tmp88[i] = 0.0
                        _tmp91 = (_tmp24[i].p - 1.0) + Y[i]
                        _tmp92 = [1.0 / _tmp91]
                        _tmp93 = zeros(size(_tmp92)) + _tmp29[i] .* (zeros(size(_tmp92 * _tmp29[i])) + _tmp90)
                        _tmp89[1] = _tmp89[1] + -(_tmp93[1]) / (_tmp91 * _tmp91)
                        _tmp75[i] = _tmp75[i] + _tmp89
                    end
                    _tmp94 = _tmp75 + _tmp67
                    _tmp95 = zeros(size(_tmp94))
                    for i = 1:length(_tmp94)
                        _tmp95[i] = (_tmp94[i])[1]
                    end
                    _tmp68[_tmp1] = _tmp68[_tmp1] + ((_tmp69 + X' * (zeros(size(_tmp13)) + exp(_tmp13) .* ((zeros(size(_tmp16)) + _tmp42 .* _tmp82) + (((zeros(size(_tmp20)) + _tmp84) + _tmp84) + -((zeros(size(_tmp21)) + _tmp95)) ./ (_tmp20 .* _tmp20))))) + _tmp72)
                    _tmp15[(_idx1 - 1) * _tmp9 + 1:_idx1 * _tmp9] = _tmp68
                end
                (_tmp3.val,_tmp10,_tmp15)
            end
        catch e # line 40:
            isa(e,Lora.LoraDSL.OutOfSupportError) || rethrow(e) # line 41:
            return (-Inf,0.0,0.0)
        end
    end

    fmod(zeros(nbeta))


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
