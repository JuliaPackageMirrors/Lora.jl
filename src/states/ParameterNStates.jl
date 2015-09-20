### Abstract parameter NStates

abstract ParameterNState{F<:VariateForm, N<:Number} <: VariableNState{F, N}

abstract ContinuousParameterNState{F<:VariateForm, N<:FloatingPoint} <: ParameterNState{F, N}

typealias MCChain ParameterNState

typealias ContinuousMCChain ContinuousParameterNState

Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterNState{F, N}}) = N
Base.eltype{F<:VariateForm, N<:FloatingPoint}(::Type{ContinuousParameterNState{F, N}}) = N

### Parameter NState subtypes

## ContinuousUnivariateParameterNState

type ContinuousUnivariateParameterNState{N<:FloatingPoint} <: ContinuousParameterNState{Univariate, N}
  value::Vector{N}
  loglikelihood::Vector{N}
  logprior::Vector{N}
  logtarget::Vector{N}
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorloglikelihood::Vector{N}
  tensorlogprior::Vector{N}
  tensorlogtarget::Vector{N}
  dtensorloglikelihood::Vector{N}
  dtensorlogprior::Vector{N}
  dtensorlogtarget::Vector{N}
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Matrix
  n::Int
  copy::Function

  ContinuousUnivariateParameterNState(
    ::Type{N},
    n::Int,
    monitor::Vector{Bool},
    diagnostickeys::Vector{Symbol}=Symbol[],
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
  ) = begin
    instance = new()

    l = Array(Int, 13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Int) : n)
    end

    for i in 1:13
      setfield!(instance, main_cpstate_fields[i], Array(N, l[i]))
    end

    instance.diagnostickeys = diagnostickeys
    instance.diagnosticvalues = diagnosticvalues
    instance.n = n

    instance.copy = eval(codegen_copy_continuous_univariate_parameter_nstate(instance, monitor))

    instance
  end
end

ContinuousUnivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
) =
  ContinuousUnivariateParameterNState{N}(N, n, monitor, diagnostickeys, diagnosticvalues)

ContinuousUnivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  n::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
) =
  ContinuousUnivariateParameterNState(
    N, n, [main_cpstate_fields[i] in monitor ? true : false for i in 1:13], diagnostickeys, diagnosticvalues
  )

typealias ContinuousUnivariateMCChain ContinuousUnivariateParameterNState

function codegen_copy_continuous_univariate_parameter_nstate(
  nstate::ContinuousUnivariateParameterNState,
  monitor::Vector{Bool}
)
  body = []
  for j in 1:13
    if monitor[j]
      push!(body, :($(nstate).(main_cpstate_fields[$j])[$(:_i)] = $(:_state).(main_cpstate_fields[$j])))
    end
  end

  if length(nstate.diagnostickeys) != 0
    push!(body, :($(nstate).diagnosticvalues[:, $(:_i)] = $(:_state).diagnosticvalues))
  end

  @gensym copy_continuous_univariate_parameter_nstate

  quote
    function $copy_continuous_univariate_parameter_nstate(_state::ContinuousUnivariateParameterState, _i::Int)
      $(body...)
    end
  end
end

Base.eltype{N<:FloatingPoint}(::Type{ContinuousUnivariateParameterNState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousUnivariateParameterNState{N}) = N

## ContinuousMultivariateParameterNState

type ContinuousMultivariateParameterNState{N<:FloatingPoint} <: ContinuousParameterNState{Multivariate, N}
  value::Matrix{N}
  loglikelihood::Vector{N}
  logprior::Vector{N}
  logtarget::Vector{N}
  gradloglikelihood::Matrix{N}
  gradlogprior::Matrix{N}
  gradlogtarget::Matrix{N}
  tensorloglikelihood::Array{N, 3}
  tensorlogprior::Array{N, 3}
  tensorlogtarget::Array{N, 3}
  dtensorloglikelihood::Array{N, 4}
  dtensorlogprior::Array{N, 4}
  dtensorlogtarget::Array{N, 4}
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Matrix
  size::Int
  n::Int
  copy::Function

  ContinuousMultivariateParameterNState(
    ::Type{N},
    size::Int,
    n::Int,
    monitor::Vector{Bool},
    diagnostickeys::Vector{Symbol}=Symbol[],
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
  ) = begin
    instance = new()

    for i in 2:4
      l = (monitor[i] == false ? zero(Int) : n)
      setfield!(instance, main_cpstate_fields[i], Array(N, l))
    end
    for i in (1, 5, 6, 7)
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_cpstate_fields[i], Array(N, s, l))
    end
    for i in 8:10
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_cpstate_fields[i], Array(N, s, s, l))
    end
    for i in 11:13
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_cpstate_fields[i], Array(N, s, s, s, l))
    end

    instance.diagnostickeys = diagnostickeys
    instance.diagnosticvalues = diagnosticvalues
    instance.size = size
    instance.n = n

    instance.copy = eval(codegen_copy_continuous_multivariate_parameter_nstate(instance, monitor))

    instance
  end
end

ContinuousMultivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
) =
  ContinuousMultivariateParameterNState{N}(N, size, n, monitor, diagnostickeys, diagnosticvalues)

ContinuousMultivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  n::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), length(diagnostickeys) == 0 ? 0 : n)
) =
  ContinuousMultivariateParameterNState(
    N, size, n, [main_cpstate_fields[i] in monitor ? true : false for i in 1:13], diagnostickeys, diagnosticvalues
  )

typealias ContinuousMultivariateMCChain ContinuousMultivariateParameterNState

function codegen_copy_continuous_multivariate_parameter_nstate(
  nstate::ContinuousMultivariateParameterNState,
  monitor::Vector{Bool}
)
  statelen::Int
  body = []

  for j in 2:4
    if monitor[j]
      push!(body, :($(nstate).(main_cpstate_fields[$j])[$(:_i)] = $(:_state).(main_cpstate_fields[$j])))
    end
  end

  for j in (1, 5, 6, 7)
    if monitor[j]
      push!(
        body,
        :(
          $(nstate).(main_cpstate_fields[$j])[1+($(:_i)-1)*$(:_state).size:$(:_i)*$(:_state).size] =
          $(:_state).(main_cpstate_fields[$j])
        )
      )
    end
  end

  if monitor[8] || monitor[9] || monitor[10]
    statelen = (nstate.size)^2
  end
  for j in 8:10
    if monitor[j]
      push!(
        body,
        :(
          $(nstate).(main_cpstate_fields[$j])[1+($(:_i)-1)*$(statelen):$(:_i)*$(statelen)] =
          $(:_state).(main_cpstate_fields[$j])
        )
      )
    end
  end

  if monitor[11] || monitor[12] || monitor[13]
    statelen = (nstate.size)^3
  end
  for j in 11:13
    if monitor[j]
      push!(
        body,
        :(
          $(nstate).(main_cpstate_fields[$j])[1+($(:_i)-1)*$(statelen):$(:_i)*$(statelen)] =
          $(:_state).(main_cpstate_fields[$j])
        )
      )
    end
  end

  if length(nstate.diagnostickeys) != 0
    push!(body, :($(nstate).diagnosticvalues[:, $(:_i)] = $(:_state).diagnosticvalues))
  end

  @gensym copy_continuous_multivariate_parameter_nstate

  quote
    function $copy_continuous_multivariate_parameter_nstate(_state::ContinuousMultivariateParameterState, _i::Int)
      $(body...)
    end
  end
end

Base.eltype{N<:FloatingPoint}(::Type{ContinuousMultivariateParameterNState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousMultivariateParameterNState{N}) = N
