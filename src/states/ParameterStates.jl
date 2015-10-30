### Abstract parameter states

abstract ParameterState{F<:VariateForm, N<:Number} <: VariableState{F, N}

abstract ContinuousParameterState{F<:VariateForm, N<:AbstractFloat} <: ParameterState{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:AbstractFloat}(::Type{ContinuousParameterState{F, N}}) = N

### Parameter state subtypes

## ContinuousUnivariateParameterState

type ContinuousUnivariateParameterState{N<:AbstractFloat} <: ContinuousParameterState{Univariate, N}
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
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Vector
end

function ContinuousUnivariateParameterState{N<:AbstractFloat}(
  value::N,
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)
  ContinuousUnivariateParameterState{N}(
    value, v, v, v, v, v, v, v, v, v, v, v, v, diagnostickeys, diagnosticvalues
  )
end

ContinuousUnivariateParameterState{N<:AbstractFloat}(
  ::Type{N},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousUnivariateParameterState(convert(N, NaN), diagnostickeys, diagnosticvalues)

Base.eltype{N<:AbstractFloat}(::Type{ContinuousUnivariateParameterState{N}}) = N
Base.eltype{N<:AbstractFloat}(s::ContinuousUnivariateParameterState{N}) = N

Base.(:(==)){S<:ContinuousUnivariateParameterState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

Base.isequal{S<:ContinuousUnivariateParameterState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])

generate_empty(state::ContinuousUnivariateParameterState) =
  ContinuousUnivariateParameterState(eltype(state), state.diagnostickeys)

## ContinuousMultivariateParameterState

type ContinuousMultivariateParameterState{N<:AbstractFloat} <: ContinuousParameterState{Multivariate, N}
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
  size::Int
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Vector
end

function ContinuousMultivariateParameterState{N<:AbstractFloat}(
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

  ContinuousMultivariateParameterState{N}(
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
    s,
    diagnostickeys,
    diagnosticvalues
  )
end

function ContinuousMultivariateParameterState{N<:AbstractFloat}(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  fnames = fieldnames(ContinuousMultivariateParameterState)
  ContinuousMultivariateParameterState(
    value, [fnames[i] in monitor ? true : false for i in 5:13], diagnostickeys, diagnosticvalues
  )
end

ContinuousMultivariateParameterState{N<:AbstractFloat}(
  ::Type{N},
  size::Int=0,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

ContinuousMultivariateParameterState{N<:AbstractFloat}(
  ::Type{N},
  size::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  ContinuousMultivariateParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

Base.eltype{N<:AbstractFloat}(::Type{ContinuousMultivariateParameterState{N}}) = N
Base.eltype{N<:AbstractFloat}(s::ContinuousMultivariateParameterState{N}) = N

Base.(:(==)){S<:ContinuousMultivariateParameterState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

Base.isequal{S<:ContinuousMultivariateParameterState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])

generate_empty(
  state::ContinuousMultivariateParameterState,
  monitor::Vector{Bool}=[isempty(fieldnames(ContinuousMultivariateParameterState)[i]) ? false : true for i in 5:13]
) =
  ContinuousMultivariateParameterState(eltype(state), state.size, monitor, state.diagnostickeys)
