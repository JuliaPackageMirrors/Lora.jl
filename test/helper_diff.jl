#########################################################################
#    Helper functions for derivation tests
#########################################################################

## error thresholds
DIFF_DELTA = 1e-9
ERROR_THRESHOLD = 2e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

##  gradient check by comparing numerical gradient to automated gradient
function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) 
	println("testing gradient of $ex at x = $x0")

	nx = length(x0)  
	myf, dummy = Abcd.generateModelFunction(ex, gradient=true, x=x0)

	pars0 = vec([x0])
	l0, grad0 = myf(pars0)  
	gradn = Array(Float64,nx)
	for i in 1:nx 
		l, grad = myf( Float64[ pars0[j] + (j==i)*DIFF_DELTA for j in 1:nx] )  
		gradn[i] = (l-l0)/DIFF_DELTA
	end

	assert( all(good_enough, zip([grad0], [gradn])),
		"Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
end


## argument pattern generation for testing
# all args can be scalar, vector or matrices, but with compatible dimensions (i.e same size for arrays)
function testpattern1(nb)
	ps = [ ifloor((i-1) / 2^(j-1)) % 2 for i=1:2^nb, j=1:nb]
	vcat(ps, 2*ps[2:end,:])
end

testpattern2(nb) = fill(0, 1, nb) # all args are scalar
testpattern3(nb) = fill(1, 1, nb) # all args are vectors
testpattern4(nb) = fill(2, 1, nb) # all args are matrices

# all args can be scalar, vector or matrices, but with no more than one array
function testpattern5(nb)
	ps = testpattern1(nb)
	ps[(ps.>0) * ones(nb) .<= 1, :]
end

## runs arg dimensions combinations
function runpattern(fex, parnames, rules, combin)
	arity = length(parnames)

	for ic in 1:size(combin,1)  # try each arg dim in combin
		c = combin[ic,:]
		par = [ symbol("arg$i") for i in 1:arity]

		# create variables
		for i in 1:arity  # generate arg1, arg2, .. variables
			vn = par[i]  
			vref = [:v0ref, :v1ref, :v2ref][c[i]+1]
			eval(:( $vn = copy($vref)))
		end

		# apply transformations on args
		for r in rules # r = rules[1]
			if isa(r, Expr) && r.head == :(->)
				pos = find(parnames .== r.args[1]) # find the arg the rules applies to 
				assert(length(pos)>0, "arg of rule ($r.args[1]) not found among $parnames")
				vn = symbol("arg$(pos[1])")
				eval(:( $vn = map($r, $vn)))
			end
		end

		# now run tests
		prange = [1:arity]
		prange = any(rules .== :(:exceptLast)) ? prange[1:end-1] : prange
		prange = any(rules .== :(:exceptFirstAndLast)) ? prange[2:end-1] : prange
		# println("$prange - $(length(rules)) - $rules")
		for p in prange  # try each argument as parameter
			tpar = copy(par)
			tpar[p] = :x  # replace tested args with parameter symbol :x for deriv1 testing func
			# f = Expr(:call, [fsym, tpar...]...) 
			f = Abcd.substSymbols(fex, Dict( parnames, tpar))
			vn = symbol("arg$p")
			x0 = eval(vn)  # set x0 for deriv 1
			deriv1(f, x0+0.001)  # shift sightly (to avoid numerical derivation pb on max() and min())
		end
	end
end

## macro to simplify tests expression
macro mtest(pattern, func::Expr, constraints...)
	tmp = [ isa(e, Symbol) ? Expr(:quote, e) : e for e in constraints]
	psym = collect( Abcd.getSymbols(func))
	quote
		local fex = $(Expr(:quote, func))
		local pars = $(Expr(:quote, [psym...]) ) 
		local rules = $(Expr(:quote, [tmp...]) ) 

		combin = ($pattern)(length(pars))
		runpattern(fex, pars, rules, combin)
	end
end
