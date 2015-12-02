### Abstract variable states

abstract VariableState{F<:VariateForm, N<:Number}

variate_form{F<:VariateForm, N<:Number}(::Type{VariableState{F, N}}) = F
Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableState{F, N}}) = N

### Basic variable state subtypes

## BasicUnvVariableState

type BasicUnvVariableState{N<:Number} <: VariableState{Univariate, N}
  value::N
end

variate_form{N<:Number}(s::Type{BasicUnvVariableState{N}}) = variate_form(super(s))
variate_form{N<:Number}(s::BasicUnvVariableState{N}) = variate_form(super(typeof(s)))

Base.eltype{N<:Number}(::Type{BasicUnvVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicUnvVariableState{N}) = N

## BasicMuvVariableState

type BasicMuvVariableState{N<:Number} <: VariableState{Multivariate, N}
  value::Vector{N}
  size::Int
end

BasicMuvVariableState{N<:Number}(value::Vector{N}) = BasicMuvVariableState{N}(value, length(value))

BasicMuvVariableState{N<:Number}(size::Int, ::Type{N}=Float64) = BasicMuvVariableState{N}(Array(N, size), size)

variate_form{N<:Number}(s::Type{BasicMuvVariableState{N}}) = variate_form(super(s))
variate_form{N<:Number}(s::BasicMuvVariableState{N}) = variate_form(super(typeof(s)))

Base.eltype{N<:Number}(::Type{BasicMuvVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicMuvVariableState{N}) = N

## BasicMavVariableState

type BasicMavVariableState{N<:Number} <: VariableState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple{Int, Int}
end

BasicMavVariableState{N<:Number}(value::Matrix{N}) = BasicMavVariableState{N}(value, size(value))

BasicMavVariableState{N<:Number}(size::Tuple, ::Type{N}=Float64) = BasicMavVariableState{N}(Array(N, size...), size)

variate_form{N<:Number}(s::Type{BasicMavVariableState{N}}) = variate_form(super(s))
variate_form{N<:Number}(s::BasicMavVariableState{N}) = variate_form(super(typeof(s)))

Base.eltype{N<:Number}(::Type{BasicMavVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicMavVariableState{N}) = N
