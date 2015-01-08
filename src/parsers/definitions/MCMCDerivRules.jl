##########################################################################################
#
#    MCMC specific derivation rules
#
##########################################################################################

####### derivation rules for Distribution constructors

for d in [Bernoulli, TDist, Exponential, Poisson]  
	typeequiv( d, 1)

	deriv_rule( d, [(:p,          Real)], :p, :( ds ) )
	deriv_rule( d, [(:p, AbstractArray)], :p, :( ds ) )
end

for d in [ Normal, Uniform, Weibull, Gamma, Cauchy, LogNormal, Binomial, Beta, Laplace]
	typeequiv( d, 2)

	deriv_rule( d, [(:p1,          Real), (:p2,          Real)], :p1, :( ds[1] ) )
	deriv_rule( d, [(:p1,          Real), (:p2,          Real)], :p2, :( ds[2] ) )

	deriv_rule( d, [(:p1, AbstractArray), (:p2, AbstractArray)], :p1, 
					quote
						res = fill(0., size(ds))
						for i in 1:length(ds)
							res[i] = ds[i][1]
						end
						res
					end )
	deriv_rule( d, [(:p1, AbstractArray), (:p2, AbstractArray)], :p2, 
					quote
						res = fill(0., size(ds))
						for i in 1:length(ds)
							res[i] = ds[i][2]
						end
						res
					end )
	# ReverseDiffSource.declareType(eval(d), d)
	# ReverseDiffSource.deriv_rule(:( ($d)(p1::Real, p2::Real) ),   :p1, :( dp1 = ds1 ) )
	# ReverseDiffSource.deriv_rule(:( ($d)(p1::Real, p2::Real) ),   :p2, :( dp2 = ds2 ) )
	# ReverseDiffSource.deriv_rule(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p1, :( copy!(dp1, ds1) ) )
	# ReverseDiffSource.deriv_rule(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p2, :( copy!(dp2, ds2) ) )
end


####### derivation rules for logpdf(Distribution, x)

macro dlogpdfx(dist::Symbol, rule) # dist = :Normal ; rule = :((d.μ - x) / (d.σ * d.σ) * ds)
	dt = eval(dist)
	deriv_rule( logpdf, [(:d,    dt), (:x,          Real)], :x, rule )

	rule2 = quote
			dx = fill(0., size(xs))
			for i in 1:length(xs)
				x = xs[i]
				dx[i] = $rule
			end
			dx
		end
	deriv_rule( logpdf, [(:d,    dt), (:xs, AbstractArray)], :xs, rule2 )

	rule3 = quote
			dx = fill(0., size(xs))
			for i in 1:length(x)
				x = xs[i]
				d = ds[i]
				dx[i] = $rule
			end
			dx
		end
	deriv_rule( logpdf, [(:ds, AbstractArray{dt}), (:xs, AbstractArray)], :xs, rule3 )

	# sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
	# ReverseDiffSource.deriv_rule( sig, :x, rule ) 

	# sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
	# rule2 = ReverseDiffSource.substSymbols(rule, {:dx => :(dx[i]), :x => :(x[i]), :ds => :(ds[i])})
	# ReverseDiffSource.deriv_rule( sig, :x, :(for i in 1:length(x) ; $rule2 ; end))

	# sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
	# rule3 = ReverseDiffSource.substSymbols(rule2, {:d => :(d[i])})
	# ReverseDiffSource.deriv_rule( sig, :x, :(for i in 1:length(x) ; $rule3 ; end))
end

macro dlogpdfd(dist::Symbol, rule)
	dt = eval(dist)
	deriv_rule( logpdf, [(:d,    dt), (:x, Real)], :d, rule )

	rule2 = quote
			dd = fill(0., size(xs))
			for i in 1:length(xs)
				x = xs[i]
				dd[i] = $rule
			end
			dd
		end
	deriv_rule( logpdf, [(:d,    dt), (:xs, AbstractArray)], :d, rule2 )

	rule3 = quote
			dd = Array(Vector, size(xs))
			for i in 1:length(x)
				x = xs[i]
				d = ds[i]
				dd[i] = $rule
			end
			dd
		end
	deriv_rule( logpdf, [(:ds, AbstractArray{dt}), (:xs, AbstractArray)], :ds, rule3 )


	# sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
	# ReverseDiffSource.deriv_rule( sig, :d, rule ) 

	# sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
	# rule2 = ReverseDiffSource.substSymbols(rule, {:x => :(x[i]), :ds => :(ds[i])})
	# ReverseDiffSource.deriv_rule( sig, :d, :(for i in 1:length(x) ; $rule2 ; end))

	# sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
	# rule2 = ReverseDiffSource.substSymbols(rule, {:dd1 => :(dd1[i]), :dd2 => :(dd2[i]), :dd3 => :(dd3[i]), 
	# 	:x => :(x[i]), :ds => :(ds[i]), :d => :(d[i]) })
	# ReverseDiffSource.deriv_rule(sig, :d, :(for i in 1:length(x) ; $rule2 ; end))
end




#######   Normal distribution
@dlogpdfx Normal (d.μ - x) / (d.σ * d.σ) * ds
@dlogpdfd Normal [ 	(x - d.μ) / (d.σ*d.σ) * ds , 
					((x - d.μ)*(x - d.μ) / (d.σ*d.σ) - 1.) / d.σ * ds ]

## Uniform distribution
@dlogpdfx Uniform 0.
@dlogpdfd Uniform [ (d.a <= x <= d.b) / (d.b - d.a) * ds ,
					(d.a <= x <= d.b) / (d.a - d.b) * ds ]

## Weibull distribution
@dlogpdfx Weibull ((1. - (x/d.β)^d.α) * d.α - 1.) / x * ds
@dlogpdfd Weibull [ ((1. - (x/d.β)^d.α) * log(x/d.β) + 1./d.α) * ds,
					((x/d.β)^d.α - 1.) * d.α/d.β * ds ]

## Beta distribution
@dlogpdfx Beta   ((d.α-1) / x - (d.β-1)/(1-x)) * ds
@dlogpdfd Beta   [	(digamma(d.α+d.β) - digamma(d.α) + log(x)) * ds ,
				    (digamma(d.α+d.β) - digamma(d.β) + log(1-x)) * ds ]


## TDist distribution
@dlogpdfx TDist		(-(d.df+1)*x / (d.df+x*x)) * ds
@dlogpdfd TDist   	((x*x-1)/(x*x + d.df)+log(d.df/(x*x+d.df))+digamma((d.df+1)/2)-digamma(d.df/2))/2 * ds

## Exponential distribution
@dlogpdfx Exponential	-ds / d.β
@dlogpdfd Exponential   (x-d.β) / (d.β*d.β) * ds

## Gamma distribution
@dlogpdfx Gamma   (-( d.β + x - d.α*d.β)/(d.β*x)) * ds
@dlogpdfd Gamma   [	(log(x) - log(d.β) - digamma(d.α)) * ds,
					((x - d.β*d.α) / (d.β*d.β)) * ds ]

## Cauchy distribution
@dlogpdfx Cauchy	(2(d.μ-x) / (d.β*d.β + (x-d.μ)*(x-d.μ))) * ds
@dlogpdfd Cauchy   	[	(2(x-d.μ) / (d.β*d.β + (x-d.μ)*(x-d.μ))) * ds,
					 	(((x-d.μ)*(x-d.μ) - d.β*d.β) / (d.β*(d.β*d.β + (x-d.μ)*(x-d.μ)))) * ds ]

## Log-normal distribution
@dlogpdfx LogNormal 	(d.nrmd.μ - d.nrmd.σ*d.nrmd.σ - log(x)) / (d.nrmd.σ*d.nrmd.σ*x) * ds 
@dlogpdfd LogNormal   	[	(log(x) - d.nrmd.μ) / (d.nrmd.σ*d.nrmd.σ) * ds ,
					 		(d.nrmd.μ*d.nrmd.μ - d.nrmd.σ*d.nrmd.σ - log(x)*(2d.nrmd.μ-log(x))) / (d.nrmd.σ*d.nrmd.σ*d.nrmd.σ) * ds ]

## Laplace distribution
@dlogpdfx Laplace	((x < d.μ)*2 - 1) / d.β * ds
@dlogpdfd Laplace   [	((x > d.μ)*2 - 1) / d.β * ds ,  # location
	                  	( abs(x-d.μ)/d.β - 2. ) / d.β * ds ]  # scale


# TODO : add other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Dirichlet, FDist
# TODO : add other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfd Bernoulli		1. / (d.p - 1. + x) * ds

## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
@dlogpdfd Binomial      [ 0. , (x / d.p - (d.n-x) / (1 - d.p)) * ds ]

## Poisson distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfd Poisson       (x / d.λ - 1) * ds


