### Abstract parameter states

abstract ParameterState{S<:ValueSupport, F<:VariateForm, N<:Number} <: VariableState{F, N}

value_support{S<:ValueSupport, F<:VariateForm, N<:Number}(::Type{ParameterState{S, F, N}}) = S
variate_form{S<:ValueSupport, F<:VariateForm, N<:Number}(::Type{ParameterState{S, F, N}}) = F
Base.eltype{S<:ValueSupport, F<:VariateForm, N<:Number}(::Type{ParameterState{S, F, N}}) = N

### Parameter state subtypes

## BasicContUnvParameterState

type BasicContUnvParameterState{N<:Real} <: ParameterState{Continuous, Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorloglikelihood::N
  tensorlogprior::N
  tensorlogtarget::N
  dtensorloglikelihood::N
  dtensorlogprior::N
  dtensorlogtarget::N
  diagnosticvalues::Vector
  diagnostickeys::Vector{Symbol}
end

function BasicContUnvParameterState{N<:Real}(
  value::N,
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)
  BasicContUnvParameterState{N}(value, v, v, v, v, v, v, v, v, v, v, v, v, diagnosticvalues, diagnostickeys)
end

BasicContUnvParameterState{N<:Real}(
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContUnvParameterState(convert(N, NaN), diagnostickeys, diagnosticvalues)

value_support{N<:Real}(s::Type{BasicContUnvParameterState{N}}) = value_support(super(s))
value_support{N<:Real}(s::BasicContUnvParameterState{N}) = value_support(super(typeof(s)))

variate_form{N<:Real}(s::Type{BasicContUnvParameterState{N}}) = variate_form(super(s))
variate_form{N<:Real}(s::BasicContUnvParameterState{N}) = variate_form(super(typeof(s)))

Base.eltype{N<:Real}(::Type{BasicContUnvParameterState{N}}) = N
Base.eltype{N<:Real}(s::BasicContUnvParameterState{N}) = N

Base.(:(==)){S<:BasicContUnvParameterState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

Base.isequal{S<:BasicContUnvParameterState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])

generate_empty(state::BasicContUnvParameterState) = BasicContUnvParameterState(state.diagnostickeys, eltype(state))

## BasicContMuvParameterState

type BasicContMuvParameterState{N<:Real} <: ParameterState{Continuous, Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorloglikelihood::Matrix{N}
  tensorlogprior::Matrix{N}
  tensorlogtarget::Matrix{N}
  dtensorloglikelihood::Array{N, 3}
  dtensorlogprior::Array{N, 3}
  dtensorlogtarget::Array{N, 3}
  diagnosticvalues::Vector
  size::Int
  diagnostickeys::Vector{Symbol}
end

function BasicContMuvParameterState{N<:Real}(
  value::Vector{N},
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)

  s = length(value)

  l = Array(Int, 9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Int) : s)
  end

  BasicContMuvParameterState{N}(
    value,
    v,
    v,
    v,
    Array(N, l[1]),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4], l[4]),
    Array(N, l[5], l[5]),
    Array(N, l[6], l[6]),
    Array(N, l[7], l[7], l[7]),
    Array(N, l[8], l[8], l[8]),
    Array(N, l[9], l[9], l[9]),
    diagnosticvalues,
    s,
    diagnostickeys
  )
end

function BasicContMuvParameterState{N<:Real}(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  fnames = fieldnames(BasicContMuvParameterState)
  BasicContMuvParameterState(value, [fnames[i] in monitor ? true : false for i in 5:13], diagnostickeys, diagnosticvalues)
end

BasicContMuvParameterState{N<:Real}(
  size::Int,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

BasicContMuvParameterState{N<:Real}(
  size::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

value_support{N<:Real}(s::Type{BasicContMuvParameterState{N}}) = value_support(super(s))
value_support{N<:Real}(s::BasicContMuvParameterState{N}) = value_support(super(typeof(s)))

variate_form{N<:Real}(s::Type{BasicContMuvParameterState{N}}) = variate_form(super(s))
variate_form{N<:Real}(s::BasicContMuvParameterState{N}) = variate_form(super(typeof(s)))

Base.eltype{N<:Real}(::Type{BasicContMuvParameterState{N}}) = N
Base.eltype{N<:Real}(s::BasicContMuvParameterState{N}) = N

Base.(:(==)){S<:BasicContMuvParameterState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

Base.isequal{S<:BasicContMuvParameterState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])

generate_empty(
  state::BasicContMuvParameterState,
  monitor::Vector{Bool}=[isempty(getfield(state, fieldnames(BasicContMuvParameterState)[i])) ? false : true for i in 5:13]
) =
  BasicContMuvParameterState(state.size, monitor, state.diagnostickeys, eltype(state))
