### Parameter NState subtypes

## ContinuousUnivariateParameterNState

type ContinuousUnivariateParameterNState{N<:FloatingPoint} <: ParameterNState{Univariate, N}
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
  diagnostics::Dict
  n::Int
  copy::Function

  ContinuousUnivariateParameterNState(::Type{N}, n::Int, monitor::Vector{Bool}, diagnostics::Dict) = begin
    instance = new()
    instance.n = n
    instance.diagnostics = diagnostics

    l = Array(Int, 13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Int) : n)
    end

    for i in 1:13
      setfield!(instance, main_state_field_names[i], Array(N, l[i]))
    end

    instance.copy = eval(codegen_copy_continuous_univariate_parameter_nstate(instance, monitor))

    instance
  end
end

ContinuousUnivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 13)],
  diagnostics::Dict=Dict()
) =
  ContinuousUnivariateParameterNState{N}(N, n, monitor, diagnostics)

ContinuousUnivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  n::Int,
  monitor::Vector{Symbol},
  diagnostics::Dict=Dict()  
) =
  ContinuousUnivariateParameterNState(
    N, n, Bool[main_state_field_names[i] in monitor ? true : false for i in 1:14], diagnostics
  )

typealias ContinuousUnivariateMCChain ContinuousUnivariateParameterNState

function codegen_copy_continuous_univariate_parameter_nstate(
  nstate::ContinuousUnivariateParameterNState,
  monitor::Vector{Bool}
)
  body = []
  for j in 1:13
    if monitor[j]
      push!(body, :($(nstate).(main_state_field_names[$j])[$(:_i)] = $(:_state).(main_state_field_names[$j])))
    end
  end

  if monitor[14]
    push!(
      body,
      :(
        for (k, v) in $(:_state).diagnostics
          if !haskey($(nstate).diagnostics, k)
            $(nstate).diagnostics[k] = Array(typeof(v), $(nstate).n)          
          end
        
          $(nstate).diagnostics[k][$(:_i)] = v
        end
      )
    )
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

type ContinuousMultivariateParameterNState{N<:FloatingPoint} <: ParameterNState{Multivariate, N}
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
  diagnostics::Dict
  size::Int
  n::Int
  copy::Function

  ContinuousMultivariateParameterNState(::Type{N}, size::Int, n::Int, monitor::Vector{Bool}, diagnostics::Dict) = begin
    instance = new()
    instance.size = size
    instance.n = n
    instance.diagnostics = diagnostics

    for i in 2:4
      l = (monitor[i] == false ? zero(Int) : n)
      setfield!(instance, main_state_field_names[i], Array(N, l))
    end
    for i in (1, 5, 6, 7)
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_state_field_names[i], Array(N, s, l))
    end
    for i in 8:10
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_state_field_names[i], Array(N, s, s, l))
    end
    for i in 11:13
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, main_state_field_names[i], Array(N, s, s, s, l))
    end

    instance.copy = eval(codegen_copy_continuous_multivariate_parameter_nstate(instance, monitor))

    instance
  end
end

ContinuousMultivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 13)],
  diagnostics::Dict=Dict()
) =
  ContinuousMultivariateParameterNState{N}(N, size, n, monitor, diagnostics)

ContinuousMultivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  size::Int,
  n::Int,
  monitor::Vector{Symbol},
  diagnostics::Dict=Dict()  
) =
  ContinuousMultivariateParameterNState(
    N, size, n, Bool[main_state_field_names[i] in monitor ? true : false for i in 1:14], diagnostics
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
      push!(body, :($(nstate).(main_state_field_names[$j])[$(:_i)] = $(:_state).(main_state_field_names[$j])))
    end
  end

  for j in (1, 5, 6, 7)
    if monitor[j]
      push!(
        body,
        :(
          $(nstate).(main_state_field_names[$j])[1+($(:_i)-1)*$(:_state).size:$(:_i)*$(:_state).size] =
          $(:_state).(main_state_field_names[$j])
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
          $(nstate).(main_state_field_names[$j])[1+($(:_i)-1)*$(statelen):$(:_i)*$(statelen)] =
          $(:_state).(main_state_field_names[$j])
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
          $(nstate).(main_state_field_names[$j])[1+($(:_i)-1)*$(statelen):$(:_i)*$(statelen)] =
          $(:_state).(main_state_field_names[$j])
        )
      )
    end
  end

  if monitor[14]
    push!(
      body,
      :(
        for (k, v) in $(:_state).diagnostics
          if !haskey($(nstate).diagnostics, k)
            $(nstate).diagnostics[k] = Array(typeof(v), $(nstate).n)          
          end
        
          $(nstate).diagnostics[k][$(:_i)] = v
        end
      )
    )
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
