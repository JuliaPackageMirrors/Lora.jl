##########################################################################
#
#    Module for DSL parsing 
#      - transforms MCMC specific idioms (~) into regular Julia syntax
#      - calls the ReverseDiffSource module for gradient code generation
#      - creates function
#
##########################################################################

module LoraDSL

	using Distributions
	using ReverseDiffSource

	import Distributions:
	  Bernoulli,
	  Beta,
	  Binomial,
	  Cauchy,
	  Exponential,
	  Gamma,
	  Laplace,
	  LogNormal,
	  Normal,
	  Poisson,
	  TDist,
	  Uniform,
	  Weibull,
	  logpdf,
	  logcdf,
	  logccdf
  
	export parsemodel

	# naming conventions
	const ACC_SYM   = :__acc     # name of accumulator variable
	const PARAM_SYM = :__beta    # name of parameter vector

	include("llacc.jl")
	include("vectordistributions.jl")
	include("diffdistributions.jl")
	include("helperfuncs.jl")
	include("parsemodel.jl")

end
