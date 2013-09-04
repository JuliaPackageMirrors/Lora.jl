#########################################################################
#    testing script for gradients
#########################################################################

using Abcd  # TODO : change name when ready

include("helper_diff.jl")

## variables of different dimension for testing
v0ref = 2.
v1ref = [2., 3, 0.1, 0, -5]
v2ref = [-1. 3 0 ; 0 5 -2]

## regular functions
@mtest testpattern1 x+y 
@mtest testpattern1 x+y+z 
@mtest testpattern1 sum(x)
@mtest testpattern1 x-y
@mtest testpattern1 x.*y
@mtest testpattern1 x./y  y -> y==0 ? 0.1 : y
@mtest testpattern1 x.^y  x -> x<=0 ? 0.2 : x 
@mtest testpattern1 sin(x)
@mtest testpattern1 abs(x)
@mtest testpattern1 cos(x)
@mtest testpattern1 exp(x)
@mtest testpattern1 log(x) x -> x<=0 ? 0.1 : x

@mtest testpattern1 transpose(x) 
deriv1(:(x'), [-3., 2, 0]) 

@mtest testpattern1 max(x,y) 
@mtest testpattern1 min(x,y)

@mtest testpattern2 x^y 

@mtest testpattern5 x/y   y->y==0 ? 0.1 : y

@mtest testpattern5 x*y 
tz = transpose(v1ref)
deriv1(:(x*tz), [-3., 2, 0]) 
deriv1(:(tz*x), v1ref)  
deriv1(:(v2ref*x), [-3., 2, 0])
deriv1(:(v2ref[:,1:2]*x), [-3. 2 0 ; 1 1 -2]) 

@mtest testpattern2 dot(x,y) 
@mtest testpattern3 dot(x,y) 

## continuous distributions
# @mtest testpattern1 logpdfNormal(mu,sigma,x)  sigma->sigma<=0?0.1:sigma
@mtest testpattern2 logpdf(Normal(mu, sigma), 2)  sigma->sigma<=0?0.1:sigma
@mtest testpattern1 logpdf(Normal(1., 0.5), x)   

# @mtest testpattern1 logpdfUniform(a,b,x)      a->a-10 b->b+10
@mtest testpattern2 logpdf(Uniform(a, b), 2)     a->a-10 b->b+10
@mtest testpattern1 logpdf(Uniform(-10, 10), x)   

# @mtest testpattern1 logpdfGamma(sh, sc,x)      sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x
@mtest testpattern2 logpdf(Gamma(sh, sc),3.)   sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc
@mtest testpattern1 logpdf(Gamma(2., 0.5), x)   x->x<=0?0.1:x

@mtest testpattern1 logpdfBeta(a,b,x)         x->clamp(x, 0.01, 0.99) a->a<=0?0.1:a b->b<=0?0.1:b
@mtest testpattern1 logpdfTDist(df,x)         df->df<=0?0.1:df
@mtest testpattern1 logpdfExponential(sc,x)   sc->sc<=0?0.1:sc  x->x<=0?0.1:x
@mtest testpattern1 logpdfCauchy(mu,sc,x)      sc->sc<=0?0.1:sc
@mtest testpattern1 logpdfLogNormal(lmu,lsc,x)  lsc->lsc<=0?0.1:lsc x->x<=0?0.1:x
@mtest testpattern1 logpdfWeibull(sh,sc,x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x

## discrete distributions
@mtest testpattern1 logpdfBernoulli(prob,x)   exceptLast prob->clamp(prob, 0.01, 0.99) x->(x>0)+0. 
# note for Bernoulli : having prob=1 or 0 is ok but will make the numeric differentiator fail => not tested

@mtest testpattern1 logpdfPoisson(l,x)   exceptLast l->l<=0?0.1:l x->iround(abs(x)) 
@mtest testpattern1 logpdfBinomial(n, prob,x)   exceptFirstAndLast prob->clamp(prob, 0.01, 0.99) x->iround(abs(x)) n->iround(abs(n)+10)


#########################################################################
#   misc. tests
#########################################################################

# Parsing should throw an error when model parameter is used as an integer variable
try
	deriv1(:(logpdfBernoulli(1, x)), [0.])
	deriv1(:(logpdfPoisson(1, x)), [0.])
	deriv1(:(logpdfBinomial(3, 0.5, x)), [0.])
	deriv1(:(logpdfBinomial(x, 0.5, 2)), [0.])
	throw("no error !!")
catch e
	assert(e != "no error !!", 
		"parser not throwing error when discrete distribution has a parameter dependant integer argument")
end

##  ref  testing
deriv1(:(x[2]),              v1ref)
deriv1(:(x[2:3]),            v1ref)
deriv1(:(x[2:end]),          v1ref)

deriv1(:(x[2:end]),          v2ref)
deriv1(:(x[2]),              v2ref)
deriv1(:(x[2:4]),            v2ref)
deriv1(:(x[:,2]),            v2ref)
deriv1(:(x[1,:]),            v2ref)
deriv1(:(x[2:end,:]),        v2ref)
deriv1(:(x[:,2:end]),        v2ref)

deriv1(:(x[2]+x[1]),          v2ref)
deriv1(:(log(x[2]^2+x[1]^2)), v2ref)

# fail case when individual elements of an array are set several times
# FIXME : correct var renaming step in unfold...
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())

