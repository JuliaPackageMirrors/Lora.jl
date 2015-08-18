### Parameter NState subtypes

## ContinuousUnivariateParameterNState

type ContinuousUnivariateParameterNState{N<:FloatingPoint} <: ParameterState{Univariate, N}
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

  ContinuousUnivariateParameterNState{N}(
    ::Type{N},
    n::Int,
    monitor::Vector{Bool}=[true, fill(false, 12)],
    diagnostics::Dict=Dict()
  ) = begin
    instance = new()
    instance.n = n

    l = Array(Int, 13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Int) : instance.n)
    end

    fields = (
      :value,
      :loglikelihood,
      :logprior,
      :logtarget,
      :gradloglikelihood,
      :gradlogprior,
      :gradlogtarget,
      :tensorloglikelihood,
      :tensorlogprior,
      :tensorlogtarget,
      :dtensorloglikelihood,
      :dtensorlogprior,
      :dtensorlogtarget
    )

    for i in 1:13
      setfield!(instance, fields[i], Array(N, l[i]))
    end

    instance.diagnostics = diagnostics
    instance.save = eval(codegen_save_continuous_univariate_parameter_nstate(instance, monitor))

    instance
  end
end

function codegen_save_continuous_univariate_parameter_nstate(
  nstate::ContinuousUnivariateParameterNState,
  monitor::Vector{Bool}
)
  fields = (
    :value,
    :loglikelihood,
    :logprior,
    :logtarget,
    :gradloglikelihood,
    :gradlogprior,
    :gradlogtarget,
    :tensorloglikelihood,
    :tensorlogprior,
    :tensorlogtarget,
    :dtensorloglikelihood,
    :dtensorlogprior,
    :dtensorlogtarget
  )

  body = {}
  for j in 1:13
    if monitor[j]
      push!(
        body,
        Expr(:(=), Expr(:ref, Expr(:., nstate, QuoteNode(fields[j])), :_i), Expr(:., :_state, QuoteNode(fields[j])))
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
