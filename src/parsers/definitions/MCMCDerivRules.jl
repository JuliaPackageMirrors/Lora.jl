##########################################################################################
#
#    MCMC specific derivation rules
#
##########################################################################################

####### derivation rules for Distribution constructors

for d in [Bernoulli, TDist, Exponential, Poisson]  
	typeequiv( d, 1)

	deriv_rule( d, [(:p,          Real)], :p, :( ds ) )
	deriv_rule( d, [(:p, AbstractArray)], :p, 
					quote
						res = zeros( size(ds) )
						for i in 1:length(ds)
							res[i] = ds[i]
						end
						res
					end )
end

for d in [ Normal, Uniform, Weibull, Gamma, Cauchy, LogNormal, Binomial, Beta, Laplace]
	typeequiv( d, 2)

	deriv_rule( d, [(:p1,          Real), (:p2,          Real)], :p1, :( ds[1] ) )
	deriv_rule( d, [(:p1,          Real), (:p2,          Real)], :p2, :( ds[2] ) )

	deriv_rule( d, [(:p1, AbstractArray), (:p2, AbstractArray)], :p1, 
					quote
						res = zeros(size(ds))
						for i in 1:length(ds)
							res[i] = ds[i][1]
						end
						res
					end )
	deriv_rule( d, [(:p1, AbstractArray), (:p2, AbstractArray)], :p2, 
					quote
						res = zeros(size(ds))
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

#### deriv rules against variable
macro dlogpdfx(dist::Symbol, rule) # dist = :Normal ; rule = :((d.μ - x) / (d.σ * d.σ) * ds)
	dt = eval(dist)
	deriv_rule( logpdf, [(:d,    dt), (:x,          Real)], :x, :( $rule * ds ) )

	rule2 = quote
			dx = zeros(size(xa))
			for i in 1:length(xa)
				x = xa[i]
				dx[i] = $rule * ds[i]
			end
			dx
		end
	deriv_rule( logpdf, [(:d,    dt), (:xa, AbstractArray)], :xa, rule2 )

	rule3 = quote
			dx = zeros(size(xa))
			for i in 1:length(xa)
				x = xa[i]
				d = da[i]
				dx[i] = $rule * ds[i]
			end
			dx
		end
	deriv_rule( logpdf, [(:da, AbstractArray{dt}), (:xa, AbstractArray)], :xa, rule3 )
end

#### deriv rules against distribution for 1 var distributions
macro dlogpdfd1(dist::Symbol, rule)
	dt = eval(dist)
	deriv_rule( logpdf, [(:d,    dt), (:x, Real)], :d, :( $rule * ds ) )

	rule2 = quote
			dd = 0.
			for i in 1:length(xa)
				x = xa[i]
				dd += $rule * ds[i]
			end
			dd
		end
	deriv_rule( logpdf, [(:d,    dt), (:xa, AbstractArray)], :d, rule2 )

	rule3 = quote
			dd = zeros(size(xa))
			for i in 1:length(xa)
				x = xa[i]
				d = da[i]
				dd[i] = $rule * ds[i]
			end
			dd
		end
	deriv_rule( logpdf, [(:da, AbstractArray{dt}), (:xa, AbstractArray)], :da, rule3 )
end

#### deriv rules against distribution for 2 to n vars distributions
macro dlogpdfd2(dist::Symbol, rule)
	dt = eval(dist)
	deriv_rule( logpdf, [(:d,    dt), (:x, Real)], :d, :( $rule .* ds ) )

	rule2 = quote
			dd = 0.
			for i in 1:length(xa)
				x = xa[i]
				dd += $rule .* ds[i]
			end
			dd
		end
	deriv_rule( logpdf, [(:d,    dt), (:xa, AbstractArray)], :d, rule2 )

	rule3 = quote
			dd = similar(xa, Vector)
			for i in 1:length(xa)
				x = xa[i]
				d = da[i]
				dd[i] = $rule .* ds[i]
			end
			dd
		end
	deriv_rule( logpdf, [(:da, AbstractArray{dt}), (:xa, AbstractArray)], :da, rule3 )
end


### Note : ... * ds term ommitted to simplify rule creation


#######   Normal distribution
@dlogpdfx Normal (d.μ - x) / (d.σ * d.σ)
@dlogpdfd2 Normal [ 	(x - d.μ) / (d.σ*d.σ) , 
					((x - d.μ)*(x - d.μ) / (d.σ*d.σ) - 1.) / d.σ ]

## Uniform distribution
@dlogpdfx Uniform 0.
@dlogpdfd2 Uniform [ (d.a <= x <= d.b) / (d.b - d.a) ,
					(d.a <= x <= d.b) / (d.a - d.b) ]

## Weibull distribution
@dlogpdfx Weibull ((1. - (x/d.β)^d.α) * d.α - 1.) / x
@dlogpdfd2 Weibull [ ((1. - (x/d.β)^d.α) * log(x/d.β) + 1./d.α),
					((x/d.β)^d.α - 1.) * d.α/d.β ]

## Beta distribution
@dlogpdfx Beta   ((d.α-1) / x - (d.β-1)/(1-x))
@dlogpdfd2 Beta   [	(digamma(d.α+d.β) - digamma(d.α) + log(x)) ,
				    (digamma(d.α+d.β) - digamma(d.β) + log(1-x)) ]

## TDist distribution
@dlogpdfx TDist		(-(d.df+1)*x / (d.df+x*x))
@dlogpdfd1 TDist   	((x*x-1)/(x*x + d.df)+log(d.df/(x*x+d.df))+digamma((d.df+1)/2)-digamma(d.df/2))/2

## Exponential distribution
@dlogpdfx Exponential   -1/d.β
@dlogpdfd1 Exponential   (x-d.β) / (d.β*d.β)

## Gamma distribution
@dlogpdfx Gamma   (-( d.β + x - d.α*d.β)/(d.β*x))
@dlogpdfd2 Gamma   [	(log(x) - log(d.β) - digamma(d.α)),
					((x - d.β*d.α) / (d.β*d.β)) ]

## Cauchy distribution
@dlogpdfx Cauchy	(2(d.μ-x) / (d.β*d.β + (x-d.μ)*(x-d.μ)))
@dlogpdfd2 Cauchy   	[	(2(x-d.μ) / (d.β*d.β + (x-d.μ)*(x-d.μ))),
					 	(((x-d.μ)*(x-d.μ) - d.β*d.β) / (d.β*(d.β*d.β + (x-d.μ)*(x-d.μ)))) ]

## Log-normal distribution
@dlogpdfx LogNormal 	(d.nrmd.μ - d.nrmd.σ*d.nrmd.σ - log(x)) / (d.nrmd.σ*d.nrmd.σ*x) 
@dlogpdfd2 LogNormal   	[	(log(x) - d.nrmd.μ) / (d.nrmd.σ*d.nrmd.σ) ,
					 		(d.nrmd.μ*d.nrmd.μ - d.nrmd.σ*d.nrmd.σ - log(x)*(2d.nrmd.μ-log(x))) / (d.nrmd.σ*d.nrmd.σ*d.nrmd.σ) ]

## Laplace distribution
@dlogpdfx Laplace	((x < d.μ)*2 - 1) / d.β
@dlogpdfd2 Laplace   [	((x > d.μ)*2 - 1) / d.β ,  # location
	                  	( abs(x-d.μ)/d.β - 2. ) / d.β ]  # scale


# TODO : add other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Dirichlet, FDist
# TODO : add other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfx Bernoulli		0.
@dlogpdfd1 Bernoulli		1. / (d.p - 1. + x)

## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
@dlogpdfx Binomial      0.
@dlogpdfd2 Binomial      [ 0. , (x / d.p - (d.n-x) / (1 - d.p)) ]

## Poisson distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfx Poisson       0.
@dlogpdfd1 Poisson       (x / d.λ - 1)


