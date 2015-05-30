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
# 3) Uplto-related fields have higher priority than implicitly derived Function tuples

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
  state::ContinuousUnivariateParameterState{N}

  ContinuousUnivariateParameter{N}(
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
    state::ContinuousUnivariateParameterState{N}
  )
    fin = tuple(setpdf!, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptdtlt)
    fnames = tuple(
      "setpdf!",
      "loglikelihood",
      "logprior",
      "logtarget",
      "gradloglikelihood",
      "gradlogprior",
      "gradlogtarget",
      "tensorloglikelihood",
      "tensorlogprior",
      "tensorlogtarget",
      "dtensorloglikelihood",
      "dtensorlogprior",
      "dtensorlogtarget",
      "uptogradlogtarget",
      "uptotensorlogtarget",
      "uptodtensorlogtarget"
    )
    nf = 16
    fout = Array(Union(Function, Nothing), nf)

    # Copy generic and anonymous functions to fout
    for i = 1:nf
      if isa(fin[i], Function)
        if isgeneric(fin[i])
          # Check that all generic functions have correct signature
          if method_exists(fin[i], (ContinuousUnivariateParameterState{N}, Dict{Variable, VariableState}))
            fout[i] = fin[i]
          else
            error("$(fnames[i]) has wrong signature")
          end
        else
          fout[i] = fin[i]
        end
      end
    end

    # Define logtarget function
    if fin[4] == nothing
      fout[4] =
        if isa(fin[2], Function) && isa(fin[3], Function)
          # pstate and nstate stand for parameter state and neighbors' state respectively
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            fin[2](pstate, nstate)+fin[3](pstate, nstate)
        elseif isa(fin[1], Function)
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            logpdf(setpdf!(pstate, nstate), pstate.value)
        elseif isa(pstate.pdf, ContinuousUnivariateDistribution) && method_exists(logpdf, (typeof(pstate.pdf), N))
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            logpdf(pstate.pdf, pstate.value)
        else
          nothing
        end
    end

    # Define gradlogtarget function
    if fin[7] == nothing
      fout[7] =
        if isa(fin[5], Function) && isa(fin[6], Function)
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            fin[5](pstate, nstate)+fin[6](pstate, nstate)
        elseif isa(fin[1], Function)
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            gradlogpdf(setpdf!(pstate, nstate), pstate.value)
        elseif isa(pstate.pdf, ContinuousUnivariateDistribution) && method_exists(logpdf, (typeof(pstate.pdf), N))
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            gradlogpdf(pstate.pdf, pstate.value)
        else
          nothing
        end
    end

    # Define tensorlogtarget function
    if fin[10] == nothing
      fout[10] =
        if isa(fin[8], Function) && isa(fin[9], Function)
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            fin[8](pstate, nstate)+fin[9](pstate, nstate)
        else
          nothing
        end
    end

    # Define dtensorlogtarget function
    if fin[13] == nothing
      fout[13] =
        if isa(fin[11], Function) && isa(fin[12], Function)
          (pstate::ContinuousUnivariateParameterState{N}, nstate::Dict{Variable, VariableState}) ->
            fin[11](pstate, nstate)+fin[12](pstate, nstate)
        else
          nothing
        end
    end

    new(index, key, distribution, fout..., state)
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
