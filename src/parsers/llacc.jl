##########################################################################################
#
#    Log likelihood accumulator : type definition and derivation rules
#
##########################################################################################
# Makes the model function easier to generate compared to a Float64
#   - embeds the error throwing when log-likelihood reaches -Inf
#   - calculates the sum when logpdf() returns an Array

type OutOfSupportError <: Exception ; end

immutable LLAcc
	val::Float64
	function LLAcc(x::Real)
		isfinite(x) || throw(OutOfSupportError())
		new(x)
	end
end
+(ll::LLAcc, x::Real)           = LLAcc(ll.val + x)
+(ll::LLAcc, x::Array{Float64}) = LLAcc(ll.val + sum(x))

####### derivation rules  ############
@deriv_rule LLAcc(x::Real)                  x      ds[1]    # constructor

# (note : only additions are possible with LLAcc type )
@deriv_rule +(x::LLAcc, y::Real)            x   ds
@deriv_rule +(x::LLAcc, y::Real)            y   ds[1]
@deriv_rule +(x::LLAcc, y::AbstractArray)   x   ds
@deriv_rule +(x::LLAcc, y::AbstractArray)   y   fill(ds[1], size(y))
