#######################################################################
#
#   parsemodel() Generates the log-likelihood function
#
#######################################################################
# - 'init' contains the dictionary of model parameters symbols and their initial value
# - 'order' specifies the derivation order (0 for derivation, 1 for gradient, 2 for Hessian, etc..)
# - 'debug' if set to true will make 'parsemodel' return the expression (instead of the function)
#       of the log-likelihood function 

function parsemodel(mex::Expr; order=0, debug=false, init...)

	# mex, order, debug, init = ex, 0, false, [(:vars, zeros(nbeta))]

	(mex.head != :block)  && (mex = Expr(:block, mex))  # enclose in block if needed

	(length(mex.args)==0) && error("model should have at least 1 statement")

	vsize, pmap, vinit = modelVars(;init...) # model param info

	mex2 = translate(mex) # rewrite ~ statements

	tn   = LLAcc.name
	llex = mexpr( tuple([fullname(tn.module)..., tn.name ]...) )
	mex2 = 	quote 
				$(ACC_SYM) = ($llex)(0.)
				$(mex2)
				$(ACC_SYM).val
			end

	dmodel = rdiff(mex2, order=order, vars=zeros(10), evalmod=Main)

	return dmodel

	


	tn   = OutOfSupportError.name
	eex  = mexpr( tuple([fullname(tn.module)..., tn.name ]...) )
	fn   = gensym("ll")
	fex = quote
			function ($fn)($(PARAM_SYM)::Vector{Float64})
				try
					$dmodel
				catch e
		          	isa(e, $eex) || rethrow(e)
		          	return tuple([-Inf, zeros($order)]...)
		        end
		    end
		end

	debug ? fex : ( current_module().eval(fex), vsize, pmap, vinit ) 
end
