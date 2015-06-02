abstract ParameterState{S<:ValueSupport, F<:VariateForm, N<:Number} <: VariableState{F, N, Random}

type ContinuousUnivariateParameterState{N<:FloatingPoint} <: ParameterState{Continuous, Univariate, N}
  value::N
  pdf::Union(ContinuousUnivariateDistribution, Nothing)
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorlogtarget::N
  dtensorlogtarget::N
end

ContinuousUnivariateParameterState{N<:FloatingPoint}(
    value::N,
    pdf::Union(ContinuousUnivariateDistribution, Nothing)=nothing
  ) =
  ContinuousUnivariateParameterState{N}(
    value,
    pdf,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN)
  )

ContinuousUnivariateParameterState{N<:FloatingPoint}(
    ::Type{N},
    pdf::Union(ContinuousUnivariateDistribution, Nothing)=nothing
  ) =
  ContinuousUnivariateParameterState(
    convert(N, NaN),
    pdf,
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN),
    convert(N, NaN)
  )

type ContinuousMultivariateParameterState{N<:FloatingPoint} <: ParameterState{Continuous, Multivariate, N}
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
  ::Type{N},
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

typealias Parameter{S<:ValueSupport, F<:VariateForm, N<:Number} Variable{F, N, Random}

# Guidelines for usage of inner constructors of continuous parameter types:
# 1) Function fields have higher priority than implicitly derived definitions via the pdf field
# 2) Target-related fields have higher priority than implicitly derived likelihood+prior fields
# 3) Upto-related fields have higher priority than implicitly derived Function tuples

type ContinuousUnivariateParameter{N<:FloatingPoint} <: Parameter{Continuous, Univariate, N}
  index::Int
  key::Symbol
  setpdf!::Union(Function, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
  gradloglikelihood::Union(Function, Nothing)
  gradlogprior::Union(Function, Nothing)
  gradlogtarget::Union(Function, Nothing)
  tensorloglikelihood::Union(Function, Nothing)
  tensorlogprior::Union(Function, Nothing)
  tensorlogtarget::Union(Function, Nothing)
  dtensorloglikelihood::Union(Function, Nothing)
  dtensorlogprior::Union(Function, Nothing)
  dtensorlogtarget::Union(Function, Nothing)
  uptogradlogtarget::Union(Function, Nothing)
  uptotensorlogtarget::Union(Function, Nothing)
  uptodtensorlogtarget::Union(Function, Nothing)
  rand::Union(Function, Nothing)
  state::ContinuousUnivariateParameterState{N}
end

function ContinuousUnivariateParameter{N<:FloatingPoint}(
  index::Int,
  key::Symbol,
  setpdf!::Union(Function, Nothing),
  ll::Union(Function, Nothing),
  lp::Union(Function, Nothing),
  lt::Union(Function, Nothing),
  gll::Union(Function, Nothing),
  glp::Union(Function, Nothing),
  glt::Union(Function, Nothing),
  tll::Union(Function, Nothing),
  tlp::Union(Function, Nothing),
  tlt::Union(Function, Nothing),
  dtll::Union(Function, Nothing),
  dtlp::Union(Function, Nothing),
  dtlt::Union(Function, Nothing),
  uptoglt::Union(Function, Nothing),
  uptotlt::Union(Function, Nothing),
  uptodtlt::Union(Function, Nothing),
  sample::Union(Function, Nothing),
  state::ContinuousUnivariateParameterState{N}
)
  functions = [
    :setpdf! => setpdf!,
    :loglikelihood => ll,
    :logprior => lp,
    :logtarget => lt,
    :gradloglikelihood => gll,
    :gradlogprior => glp,
    :gradlogtarget => glt,
    :tensorloglikelihood => tll,
    :tensorlogprior => tlp,
    :tensorlogtarget => tlt,
    :dtensorloglikelihood => dtll,
    :dtensorlogprior => dtlp,
    :dtensorlogtarget => dtlt,
    :uptogradlogtarget => uptoglt,
    :uptotensorlogtarget => uptotlt,
    :uptodtensorlogtarget => uptodtlt,
    :rand => sample
  ]

  # Check that all generic functions have correct signature
  for (k, f) in functions
    if isgeneric(f) && !method_exists(f, (ContinuousUnivariateParameterState{N}, Dict{Variable, VariableState}))
      error("$k has wrong signature")
    end
  end

  # Define logtarget and gradlogtarget
  for (kt, kl, kp, f) in (
    (:logtarget, :loglikelihood, :logprior, logpdf),
    (:gradlogtarget, :gradloglikelihood, :gradlogprior, gradlogpdf)
  )
    if functions[kt] == nothing
      if isa(functions[kl], Function) && isa(functions[kp], Function)
        # pstate and nstate stand for parameter state and neighbors' state respectively
        functions[kt] = 
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
          functions[kl](pstate, nstate)+functions[kp](pstate, nstate)
      elseif isa(functions[:setpdf!], Function)
        functions[kt] =
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
          f(setpdf!(pstate, nstate), pstate.value)
      elseif isa(state.pdf, ContinuousUnivariateDistribution) && method_exists(f, (typeof(state.pdf), N))
        functions[kt] =
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            f(pstate.pdf, pstate.value)
      end
    end
  end

  # Define tensorlogtarget and dtensorlogtarget
  for (kt, kl, kp) in (
    (:tensorlogtarget, :tensorloglikelihood, :tensorlogprior),
    (:dtensorlogtarget, :dtensorloglikelihood, :dtensorlogprior)
  )
    if functions[kt] == nothing && isa(functions[kl], Function) && isa(functions[kp], Function)
      functions[kt] =
        (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
        functions[kl](pstate, nstate)+functions[kp](pstate, nstate)
    end
  end

  # Define uptogradlogtarget
  if functions[:uptogradlogtarget] == nothing &&
    isa(functions[:logtarget], Function) &&
    isa(functions[:gradlogtarget], Function)
    functions[:uptogradlogtarget] =
      (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
      (functions[:logtarget](pstate, nstate), functions[:gradlogtarget](pstate, nstate))
  end

  # Define uptotensorlogtarget
  if functions[:uptotensorlogtarget] == nothing &&
    isa(functions[:logtarget], Function) &&
    isa(functions[:gradlogtarget], Function) &&
    isa(functions[:tensorlogtarget], Function)   
    functions[:uptotensorlogtarget] =
      (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
      (
        functions[:logtarget](pstate, nstate),
        functions[:gradlogtarget](pstate, nstate),
        functions[:tensorlogtarget](pstate, nstate)
      )
  end

  # Define uptodtensorlogtarget
  if functions[:uptodtensorlogtarget] == nothing &&
    isa(functions[:logtarget], Function) &&
    isa(functions[:gradlogtarget], Function) &&
    isa(functions[:tensorlogtarget], Function) &&
    isa(functions[:dtensorlogtarget], Function)
    functions[:uptodtensorlogtarget] =
      (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
      (
        functions[:logtarget](pstate, nstate),
        functions[:gradlogtarget](pstate, nstate),
        functions[:tensorlogtarget](pstate, nstate),
        functions[:dtensorlogtarget](pstate, nstate)
      )
  end

  # Define rand
  if functions[:rand] == nothing
    if isa(functions[:setpdf!], Function)
      functions[:rand] = 
        (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
        rand(setpdf!(pstate, nstate))
    elseif isa(state.pdf, ContinuousUnivariateDistribution) && method_exists(rand, (typeof(state.pdf),))
      functions[:rand] = 
        (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}, n::Int) ->
        rand(pstate.pdf)
    end
  end

  ContinuousUnivariateParameter{N}(
    index,
    key,
    functions[:setpdf!],
    functions[:loglikelihood],
    functions[:logprior],
    functions[:logtarget],
    functions[:gradloglikelihood],
    functions[:gradlogprior],
    functions[:gradlogtarget],
    functions[:tensorloglikelihood],
    functions[:tensorlogprior],
    functions[:tensorlogtarget],
    functions[:dtensorloglikelihood],
    functions[:dtensorlogprior],
    functions[:dtensorlogtarget],
    functions[:uptogradlogtarget],
    functions[:uptotensorlogtarget],
    functions[:uptodtensorlogtarget],
    functions[:rand],
    state
  )
end

type ContinuousMultivariateParameter{N<:FloatingPoint} <: Parameter{Continuous, Multivariate, N}
  index::Int
  key::Symbol
  distribution::Union(ContinuousMultivariateDistribution, Nothing)
  loglikelihood::Union(Function, Nothing)
  logprior::Union(Function, Nothing)
  logtarget::Union(Function, Nothing)
  gradloglikelihood::Union(Function, Nothing)
  gradlogprior::Union(Function, Nothing)
  gradlogtarget::Union(Function, Nothing)
  tensorloglikelihood::Union(Function, Nothing)
  tensorlogprior::Union(Function, Nothing)
  tensorlogtarget::Union(Function, Nothing)
  dtensorloglikelihood::Union(Function, Nothing)
  dtensorlogprior::Union(Function, Nothing)
  dtensorlogtarget::Union(Function, Nothing)
  uptogradlogtarget::Union(Function, Nothing)
  uptotensorlogtarget::Union(Function, Nothing)
  uptodtensorlogtarget::Union(Function, Nothing)
  state::ContinuousMultivariateParameterState{N}
end
