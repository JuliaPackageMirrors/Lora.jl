using Distributions
using DataFrames

pwd()
cd("MCMC.jl/src")
include("MCMC.jl") 
# using MCMC

# simulate dataset
srand(1)
nbeta = 10 # number of predictors, including intercept
beta0 = randn((nbeta,))

n = 1000
X = [ones(n) randn((n, nbeta-1))]
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

MCMC.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)

#####  version ref #####
	let 
        global _ll1
        local X = Main.X
        local Y = Main.Y
        local __tmp_258 = -(1.0,Y)
        local __tmp_261 = -(1)
        local __tmp_265 = transpose(X)
        local __tmp_267 = .*(1.0,1.0)
        function _ll1(__beta::Vector{Float64})
            try 
                local vars = __beta[1:10]
                local __acc = 0.0
                local d____acc_256 = 1.0
                local d__tmp_250 = 0.0
                local d__tmp_251 = zeros(Float64,(n,))
                local d__tmp_253 = zeros(Float64,(n,))
                local d__tmp_254 = 0.0
                local d__tmp_252 = zeros(Float64,(n,))
                local dvars = zeros(Float64,(10,))
                local dprob = zeros(Float64,(n,))
                local d____acc_255 = 0.0
                local __tmp_250 = MCMC.logpdfNormal(0,1.0,vars)
                local ____acc_255 = +(__acc,__tmp_250)
                local __tmp_251 = *(X,vars)
                local __tmp_252 = exp(__tmp_251)
                local __tmp_253 = +(1.0,__tmp_252)
                local prob = /(1,__tmp_253)
                local __tmp_254 = MCMC.logpdfBernoulli(prob,Y)
                local ____acc_256 = +(____acc_255,__tmp_254)
                d____acc_255 = +(d____acc_255,sum(d____acc_256))
                d__tmp_254 = +(d__tmp_254,sum(d____acc_256))
                local __tmp_259 = -(prob,__tmp_258)
                local __tmp_260 = ./(1.0,__tmp_259)
                dprob = +(dprob,*(__tmp_260,d__tmp_254))
                local __tmp_262 = .*(__tmp_253,__tmp_253)
                local __tmp_263 = ./(__tmp_261,__tmp_262)
                d__tmp_253 = +(d__tmp_253,.*(__tmp_263,dprob))
                d__tmp_252 = +(d__tmp_252,+(d__tmp_253))
                local __tmp_264 = exp(__tmp_251)
                d__tmp_251 = +(d__tmp_251,.*(__tmp_264,d__tmp_252))
                dvars = +(dvars,*(__tmp_265,d__tmp_251))
                d__tmp_250 = +(d__tmp_250,sum(d____acc_255))
                local __tmp_266 = -(0,vars)
                local __tmp_268 = ./(__tmp_266,__tmp_267)
                dvars = +(dvars,*(__tmp_268,d__tmp_250))
                local d__beta = similar(__beta)
                d__beta[1:10] = dvars
                (____acc_256,d__beta)
            catch e
                if (e=="give up eval") 
                    return (-(Inf),zero(__beta))
                else 
                    throw(e)
                end
            end
        end
    end


#####  version optim #####
	let 
        global _ll2
        local X = Main.X
        local Y = Main.Y
        local __tmp_258 = -(1.0,Y)
        local __tmp_261 = -(1)
        local __tmp_265 = transpose(X)
        local __tmp_267 = .*(1.0,1.0)
 
        local d__tmp_251 = Array(Float64,(n,))
        local d__tmp_253 = Array(Float64,(n,))
        local d__tmp_252 = Array(Float64,(n,))
        local dvars = Array(Float64,(10,))
        local dprob = Array(Float64,(n,))
        local prob = Array(Float64,(n,))
        local __tmp_251 = Array(Float64,(n,))
        local __tmp_253 = Array(Float64,(n,))
        local __tmp_262 = Array(Float64,(n,))
        local __tmp_260 = Array(Float64,(n,))
        local __tmp_259 = Array(Float64,(n,))
        local __tmp_263 = Array(Float64,(n,))

        function _ll2(__beta::Vector{Float64})
            try 
                local vars = __beta[1:10]
                local __acc = 0.0
                local d____acc_256 = 1.0
                local d__tmp_250 = 0.0
                local d__tmp_254 = 0.0
                local d____acc_255 = 0.0

                fill!(d__tmp_251, 0.)
                fill!(d__tmp_253, 0.)
                fill!(d__tmp_252, 0.)
                fill!(dvars, 0.)
                fill!(dprob, 0.)

                local __tmp_250 = MCMC.logpdfNormal(0,1.0,vars)
                local ____acc_255 = +(__acc,__tmp_250) #  scalar only
                
                gemm!('N', 'N', 1., X, reshape(vars,10,1), 0., __tmp_251) #   = *(X,vars)
                
                exp!(__tmp_251)
                map!(Add(), __tmp_253, 1., __tmp_251)
                map!(Divide(), prob, 1., __tmp_253)
                local __tmp_254 = MCMC.logpdfBernoulli(prob,Y)
                local ____acc_256 = +(____acc_255,__tmp_254) #  scalar only

                d____acc_255 = +(d____acc_255, d____acc_256) #  scalar only
                d__tmp_254 = +(d__tmp_254, d____acc_256)   #  scalar only

                # local __tmp_259 = -(prob,__tmp_258)
                map!(Subtract(), __tmp_259, prob,__tmp_258)
                # local __tmp_260 = ./(1.0,__tmp_259)
                # dprob = +(dprob,*(__tmp_260,d__tmp_254))
                map!(Divide(), __tmp_260, d__tmp_254, __tmp_259)
                map1!(Add(), dprob, __tmp_260)

                # local __tmp_262 = .*(__tmp_253,__tmp_253)
                map!(Multiply(), __tmp_262, __tmp_253, __tmp_253)
                # local __tmp_263 = ./(__tmp_261,__tmp_262)
                map!(Divide(), __tmp_263, __tmp_261, __tmp_262)

                # d__tmp_253 = +(d__tmp_253,.*(__tmp_263,dprob))
                map1!(Multiply(), __tmp_263, dprob)
                map1!(Add(), d__tmp_253, __tmp_263)

                # d__tmp_252 = +(d__tmp_252,+(d__tmp_253))
                map1!(Add(), d__tmp_252, d__tmp_253)

                #local __tmp_264 = exp(__tmp_251)   # doublon

                # d__tmp_251 = +(d__tmp_251,.*(__tmp_251,d__tmp_252))  
                map1!(Multiply(), __tmp_251, d__tmp_252)
                map1!(Add(), d__tmp_251, __tmp_251)

                dvars = +(dvars,*(__tmp_265,d__tmp_251))

                d__tmp_250 = +(d__tmp_250, d____acc_255)

                local __tmp_266 = -(0,vars)
                divide!(__tmp_266,__tmp_267)
                dvars = +(dvars,*(__tmp_266,d__tmp_250))

                local d__beta = similar(__beta)
                d__beta[1:10] = dvars
                (____acc_256,d__beta)
            catch e
                if (e=="give up eval") 
                    return (-(Inf),zero(__beta))
                else 
                    throw(e)
                end
            end
        end
    end


_ll1(zeros(nbeta))

@time begin 
		s = 0.
		for i in 1:1000
			t = randn(nbeta)
			s += _ll1(t)[1]
		end
	end  # 0.75 sec

using Base.LinAlg.BLAS
using NumericExtensions

gemm!

_ll2(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:1000
			t = randn(nbeta)
			s += _ll2(t)[1]
		end
	end  # 0.22 sec


##############  avec n = 100_000  #####################

n = 100000
X = [ones(n) randn((n, nbeta-1))]
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))


_ll1(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:10
			t = randn(nbeta)
			s += _ll1(t)[1]
		end
	end  # 0.62 sec


_ll2(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:10
			t = randn(nbeta)
			s += _ll2(t)[1]
		end
	end  # 0.23 sec
