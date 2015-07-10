abstract VariableState{F<:VariateForm, N<:Number}

abstract GenericVariableState{F<:VariateForm, N<:Number} <: VariableState{F, N}

type UnivariateGenericVariableState{N<:Number} <: GenericVariableState{Univariate, N}
  value::N
end

type MultivariateGenericVariableState{N<:Number} <: GenericVariableState{Multivariate, N}
  value::Vector{N}
  size::Int
end

type MatrixvariateGenericVariableState{N<:Number} <: GenericVariableState{Matrixvariate, N}
  value::Matrix{N}
  size::Tuple
end

abstract ParameterState{F<:VariateForm, N<:Number} <: VariableState{F, N}

type ContinuousUnivariateParameterState{N<:FloatingPoint} <: ParameterState{Univariate, N}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorlogtarget::N
  dtensorlogtarget::N
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(value::N) =
  ContinuousUnivariateParameterState{N}(
    value,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN)
  )

ContinuousUnivariateParameterState{N<:FloatingPoint}(::Type{N}=Float64) =
  ContinuousUnivariateParameterState(
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN)
  )

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ParameterState{Multivariate, N}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorlogtarget::Matrix{N}
  dtensorlogtarget::Array{N, 3}
  size::Int
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(value::Vector{N}, monitor::Vector{Bool}=fill(false, 5))
  size::Int = length(value)

  l::Vector{Int} = Array(Int, 5)
  for i in 1:5
    l[i] = (monitor[i] == false ? zero(Int) : size)
  end

  ContinuousMultivariateParameterState{N}(
    value,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    Array(N, l[1]),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4], l[4]),
    Array(N, l[5], l[5], l[5]),
    size
  )
end

function ContinuousMultivariateParameterState{N<:FloatingPoint}(
  ::Type{N}=Float64,
  size::Int=0,
  monitor::Vector{Bool}=[true, fill(false, 6)]
  )

  l::Vector{Int} = Array(Int, 6)
  for i in 1:6
    l[i] = (monitor[i] == false ? zero(Int) : size)
  end

  ContinuousMultivariateParameterState{N}(
    Array(N, l[1]),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4]),
    Array(N, l[5], l[5]),
    Array(N, l[6], l[6], l[6]),
    l[1]
  )
end
