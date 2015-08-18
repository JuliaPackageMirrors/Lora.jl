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
  save::Function

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

    instance.save = eval(codegen_save_continuous_univariate_parameter_nstate(instance, monitor))

    instance
  end
end

ContinuousUnivariateParameterNState{N<:FloatingPoint}(
  ::Type{N},
  n::Int,
  monitor::Vector{Bool}=[true, fill(false, 13)],
  diagnostics::Dict=Dict()
) =
  ContinuousUnivariateParameterNState{N}(N, n, monitor, diagnostics)

typealias ContinuousUnivariateMCChain ContinuousUnivariateParameterNState

function codegen_save_continuous_univariate_parameter_nstate(
  nstate::ContinuousUnivariateParameterNState,
  monitor::Vector{Bool}
)
  body = {}
  for j in 1:13
    if monitor[j]
      push!(
        body,
        :(
          $(nstate).(main_state_field_names[$j])[$(:_i)] = $(:_state).(main_state_field_names[$j])
        )
      )
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
  end

  @gensym save_continuous_univariate_parameter_nstate

  quote
    function $save_continuous_univariate_parameter_nstate(_state::ContinuousUnivariateParameterState, _i::Int)
      $(body...)
    end
  end
end

Base.eltype{N<:FloatingPoint}(::Type{ContinuousUnivariateParameterNState{N}}) = N
Base.eltype{N<:FloatingPoint}(s::ContinuousUnivariateParameterNState{N}) = N
