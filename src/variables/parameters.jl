### Astract Parameter types

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

### Parameter subtypes

## ContinuousUnivariateParameter

# Guidelines for usage of inner constructors of continuous parameter types:
# 1) Function fields have higher priority than implicitly derived definitions via the pdf field
# 2) Target-related fields have higher priority than implicitly derived likelihood+prior fields
# 3) Upto-related fields have higher priority than implicitly derived Function tuples

type ContinuousUnivariateParameter <: Parameter{Continuous, Univariate}
  index::Int
  key::Symbol
  pdf::Union(ContinuousUnivariateDistribution, Nothing)
  prior::Union(ContinuousUnivariateDistribution, Nothing)
  setpdf::Union(Function, Nothing)
  setprior::Union(Function, Nothing)
  loglikelihood!::Union(Function, Nothing)
  logprior!::Union(Function, Nothing)
  logtarget!::Union(Function, Nothing)
  gradloglikelihood!::Union(Function, Nothing)
  gradlogprior!::Union(Function, Nothing)
  gradlogtarget!::Union(Function, Nothing)
  tensorloglikelihood!::Union(Function, Nothing)
  tensorlogprior!::Union(Function, Nothing)
  tensorlogtarget!::Union(Function, Nothing)
  dtensorloglikelihood!::Union(Function, Nothing)
  dtensorlogprior!::Union(Function, Nothing)
  dtensorlogtarget!::Union(Function, Nothing)
  uptogradlogtarget!::Union(Function, Nothing)
  uptotensorlogtarget!::Union(Function, Nothing)
  uptodtensorlogtarget!::Union(Function, Nothing)

  ContinuousUnivariateParameter(
    index::Int,
    key::Symbol,
    pdf::Union(ContinuousUnivariateDistribution, Nothing),
    prior::Union(ContinuousUnivariateDistribution, Nothing),
    setpdf::Union(Function, Nothing),
    setprior::Union(Function, Nothing),
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
    uptodtlt::Union(Function, Nothing)
  ) = begin
    instance = new()
    instance.index = index
    instance.key = key
    instance.pdf = pdf
    instance.prior = prior

    fin = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = (
      "setpdf",
      "setprior",
      "loglikelihood!",
      "logprior!",
      "logtarget!",
      "gradloglikelihood!",
      "gradlogprior!",
      "gradlogtarget!",
      "tensorloglikelihood!",
      "tensorlogprior!",
      "tensorlogtarget!",
      "dtensorloglikelihood!",
      "dtensorlogprior!",
      "dtensorlogtarget!",
      "uptogradlogtarget!",
      "uptotensorlogtarget!",
      "uptodtensorlogtarget!"
    )
    nf = 17

    # Check that all generic functions have correct signature
    for i = 1:nf
      if isa(fin[i], Function) &&
        isgeneric(fin[i]) &&
        !method_exists(fin[i], (ContinuousUnivariateParameterState, Dict{Symbol, VariableState}))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(fin[i], Function)
          # pstate and nstate stand for parameter state and neighbors' state respectively
          (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
          setfield!(instance, distribution, fin[i](pstate, nstate))
        else
          fin[i]
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = fin[3]
    instance.gradloglikelihood! = fin[6]

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf),
      (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if fin[i] == nothing && (
          (isa(prior, ContinuousUnivariateDistribution) && method_exists(f, (typeof(prior), FloatingPoint))) ||
          isa(fin[2], Function)
        )
          (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
          setfield!(pstate, spfield, f(instance.prior, pstate.value))
        else
          fin[i]
        end
      )
    end

    # Define logtarget! (i = 5) and gradlogtarget! (i = 8)
    # ptfield, plfield and ppfield stand for parameter target, likelihood and prior-related field respectively
    # stfield, slfield and spfield stand for state target, likelihood and prior-related field respectively
    for (i , ptfield, plfield, ppfield, stfield, slfield, spfield, f) in (
      (
        5,
        :logtarget!, :loglikelihood!, :logprior!,
        :logtarget, :loglikelihood, :logprior,
        logpdf
      ),
      (
        8,
        :gradlogtarget!, :gradloglikelihood!, :gradlogprior!,
        :gradlogtarget, :gradloglikelihood, :gradlogprior,
        gradlogpdf
      )
    )
      setfield!(
        instance,
        ptfield,
        if fin[i] == nothing
          if isa(fin[i-2], Function) && isa(getfield(instance, pfield), Function)
            function (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState})
              getfield(instance, plfield)(pstate, nstate)
              getfield(instance, ppfield)(pstate, nstate)
              setfield!(pstate, stfield, getfield(pstate, slfield)+getfield(pstate, spfield))
            end
          elseif (isa(pdf, ContinuousUnivariateDistribution) && method_exists(f, (typeof(pdf), FloatingPoint))) ||
            isa(fin[1], Function)
            (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
            setfield!(pstate, stfield, f(instance.pdf, pstate.value))
          end
        else
          fin[i]
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = fin[9]
    instance.dtensorloglikelihood! = fin[12]

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = fin[10]
    instance.dtensorlogprior! = fin[13]

    # Define tensorlogtarget! (i = 11) and dtensorlogtarget! (i = 14)
    for (i , ptfield, plfield, ppfield, stfield, slfield, spfield) in (
      (
        11,
        :tensorlogtarget!, :tensorloglikelihood!, :tensorlogprior!,
        :tensorlogtarget, :tensorloglikelihood, :tensorlogprior
      ),
      (
        14,
        :dtensorlogtarget!, :dtensorloglikelihood!, :dtensorlogprior!,
        :dtensorlogtarget, :dtensorloglikelihood, :dtensorlogprior
      )
    )
      setfield!(
        instance,
        ptfield,
        if fin[i] == nothing && isa(fin[i-2], Function) && isa(fin[i-1], Function)
          function (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState})
            getfield(instance, plfield)(pstate, nstate)
            getfield(instance, ppfield)(pstate, nstate)
            setfield!(pstate, stfield, getfield(pstate, slfield)+getfield(pstate, spfield))
          end
        else
          fin[i]
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if fin[15] == nothing && isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
        function (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
        end
      else
        fin[15]
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if fin[16] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function)
        function (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
          instance.tensorlogtarget!(pstate, nstate)          
        end
      else
        fin[16]
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if fin[17] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function) &&
        isa(instance.dtensorlogtarget!, Function)
        function (pstate::ContinuousUnivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
          instance.tensorlogtarget!(pstate, nstate)
          instance.dtensorlogtarget!(pstate, nstate)
        end
      else
        fin[17]
      end
    )

    instance
  end
end

function ContinuousUnivariateParameter(
  index::Int,
  key::Symbol;
  pdf::Union(ContinuousUnivariateDistribution, Nothing)=nothing,
  prior::Union(ContinuousUnivariateDistribution, Nothing)=nothing,  
  setpdf::Union(Function, Nothing)=nothing,
  setprior::Union(Function, Nothing)=nothing,
  loglikelihood::Union(Function, Nothing)=nothing,
  logprior::Union(Function, Nothing)=nothing,
  logtarget::Union(Function, Nothing)=nothing,
  gradloglikelihood::Union(Function, Nothing)=nothing,
  gradlogprior::Union(Function, Nothing)=nothing,
  gradlogtarget::Union(Function, Nothing)=nothing,
  tensorloglikelihood::Union(Function, Nothing)=nothing,
  tensorlogprior::Union(Function, Nothing)=nothing,
  tensorlogtarget::Union(Function, Nothing)=nothing,
  dtensorloglikelihood::Union(Function, Nothing)=nothing,
  dtensorlogprior::Union(Function, Nothing)=nothing,
  dtensorlogtarget::Union(Function, Nothing)=nothing,
  uptogradlogtarget::Union(Function, Nothing)=nothing,
  uptotensorlogtarget::Union(Function, Nothing)=nothing,
  uptodtensorlogtarget::Union(Function, Nothing)=nothing
)
  ContinuousUnivariateParameter(
    index,
    key,
    pdf,
    prior,
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )
end

## ContinuousMultivariateParameter

type ContinuousMultivariateParameter <: Parameter{Continuous, Univariate}
  index::Int
  key::Symbol
  pdf::Union(ContinuousMultivariateDistribution, Nothing)
  prior::Union(ContinuousMultivariateDistribution, Nothing)
  setpdf::Union(Function, Nothing)
  setprior::Union(Function, Nothing)
  loglikelihood!::Union(Function, Nothing)
  logprior!::Union(Function, Nothing)
  logtarget!::Union(Function, Nothing)
  gradloglikelihood!::Union(Function, Nothing)
  gradlogprior!::Union(Function, Nothing)
  gradlogtarget!::Union(Function, Nothing)
  tensorloglikelihood!::Union(Function, Nothing)
  tensorlogprior!::Union(Function, Nothing)
  tensorlogtarget!::Union(Function, Nothing)
  dtensorloglikelihood!::Union(Function, Nothing)
  dtensorlogprior!::Union(Function, Nothing)
  dtensorlogtarget!::Union(Function, Nothing)
  uptogradlogtarget!::Union(Function, Nothing)
  uptotensorlogtarget!::Union(Function, Nothing)
  uptodtensorlogtarget!::Union(Function, Nothing)

  ContinuousMultivariateParameter(
    index::Int,
    key::Symbol,
    pdf::Union(ContinuousMultivariateDistribution, Nothing),
    prior::Union(ContinuousMultivariateDistribution, Nothing),
    setpdf::Union(Function, Nothing),
    setprior::Union(Function, Nothing),
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
    uptodtlt::Union(Function, Nothing)
  ) = begin
    instance = new()
    instance.index = index
    instance.key = key
    instance.pdf = pdf
    instance.prior = prior

    fin = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = (
      "setpdf",
      "setprior",
      "loglikelihood!",
      "logprior!",
      "logtarget!",
      "gradloglikelihood!",
      "gradlogprior!",
      "gradlogtarget!",
      "tensorloglikelihood!",
      "tensorlogprior!",
      "tensorlogtarget!",
      "dtensorloglikelihood!",
      "dtensorlogprior!",
      "dtensorlogtarget!",
      "uptogradlogtarget!",
      "uptotensorlogtarget!",
      "uptodtensorlogtarget!"
    )
    nf = 17

    # Check that all generic functions have correct signature
    for i = 1:nf
      if isa(fin[i], Function) &&
        isgeneric(fin[i]) &&
        !method_exists(fin[i], (ContinuousMultivariateParameterState, Dict{Symbol, VariableState}))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(fin[i], Function)
          # pstate and nstate stand for parameter state and neighbors' state respectively
          (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
          setfield!(instance, distribution, fin[i](pstate, nstate))
        else
          fin[i]
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = fin[3]
    instance.gradloglikelihood! = fin[6]

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf),
      (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if fin[i] == nothing && (
          (
            isa(prior, ContinuousMultivariateDistribution) && 
            method_exists(f, (typeof(prior), Vector{FloatingPoint}))
          ) ||
          isa(fin[2], Function)
        )
          (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
          setfield!(pstate, spfield, f(instance.prior, pstate.value))
        else
          fin[i]
        end
      )
    end

    # Define logtarget! (i = 5) and gradlogtarget! (i = 8)
    # ptfield, plfield and ppfield stand for parameter target, likelihood and prior-related field respectively
    # stfield, slfield and spfield stand for state target, likelihood and prior-related field respectively
    for (i , ptfield, plfield, ppfield, stfield, slfield, spfield, f) in (
      (
        5,
        :logtarget!, :loglikelihood!, :logprior!,
        :logtarget, :loglikelihood, :logprior,
        logpdf
      ),
      (
        8,
        :gradlogtarget!, :gradloglikelihood!, :gradlogprior!,
        :gradlogtarget, :gradloglikelihood, :gradlogprior,
        gradlogpdf
      )
    )
      setfield!(
        instance,
        ptfield,
        if fin[i] == nothing
          if isa(fin[i-2], Function) && isa(getfield(instance, pfield), Function)
            function (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState})
              getfield(instance, plfield)(pstate, nstate)
              getfield(instance, ppfield)(pstate, nstate)
              setfield!(pstate, stfield, getfield(pstate, slfield)+getfield(pstate, spfield))
            end
          elseif (
              isa(pdf, ContinuousMultivariateDistribution) &&
              method_exists(f, (typeof(pdf), Vector{FloatingPoint}))
            ) ||
            isa(fin[1], Function)
            (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState}) ->
            setfield!(pstate, stfield, f(instance.pdf, pstate.value))
          end
        else
          fin[i]
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = fin[9]
    instance.dtensorloglikelihood! = fin[12]

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = fin[10]
    instance.dtensorlogprior! = fin[13]

    # Define tensorlogtarget! (i = 11) and dtensorlogtarget! (i = 14)
    for (i , ptfield, plfield, ppfield, stfield, slfield, spfield) in (
      (
        11,
        :tensorlogtarget!, :tensorloglikelihood!, :tensorlogprior!,
        :tensorlogtarget, :tensorloglikelihood, :tensorlogprior
      ),
      (
        14,
        :dtensorlogtarget!, :dtensorloglikelihood!, :dtensorlogprior!,
        :dtensorlogtarget, :dtensorloglikelihood, :dtensorlogprior
      )
    )
      setfield!(
        instance,
        ptfield,
        if fin[i] == nothing && isa(fin[i-2], Function) && isa(fin[i-1], Function)
          function (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState})
            getfield(instance, plfield)(pstate, nstate)
            getfield(instance, ppfield)(pstate, nstate)
            setfield!(pstate, stfield, getfield(pstate, slfield)+getfield(pstate, spfield))
          end
        else
          fin[i]
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if fin[15] == nothing && isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
        function (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
        end
      else
        fin[15]
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if fin[16] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function)
        function (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
          instance.tensorlogtarget!(pstate, nstate)          
        end
      else
        fin[16]
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if fin[17] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function) &&
        isa(instance.dtensorlogtarget!, Function)
        function (pstate::ContinuousMultivariateParameterState, nstate::Dict{Symbol, VariableState})
          instance.logtarget!(pstate, nstate)
          instance.gradlogtarget!(pstate, nstate)
          instance.tensorlogtarget!(pstate, nstate)
          instance.dtensorlogtarget!(pstate, nstate)
        end
      else
        fin[17]
      end
    )

    instance
  end
end

function ContinuousMultivariateParameter(
  index::Int,
  key::Symbol;
  pdf::Union(ContinuousMultivariateDistribution, Nothing)=nothing,
  prior::Union(ContinuousMultivariateDistribution, Nothing)=nothing,  
  setpdf::Union(Function, Nothing)=nothing,
  setprior::Union(Function, Nothing)=nothing,
  loglikelihood::Union(Function, Nothing)=nothing,
  logprior::Union(Function, Nothing)=nothing,
  logtarget::Union(Function, Nothing)=nothing,
  gradloglikelihood::Union(Function, Nothing)=nothing,
  gradlogprior::Union(Function, Nothing)=nothing,
  gradlogtarget::Union(Function, Nothing)=nothing,
  tensorloglikelihood::Union(Function, Nothing)=nothing,
  tensorlogprior::Union(Function, Nothing)=nothing,
  tensorlogtarget::Union(Function, Nothing)=nothing,
  dtensorloglikelihood::Union(Function, Nothing)=nothing,
  dtensorlogprior::Union(Function, Nothing)=nothing,
  dtensorlogtarget::Union(Function, Nothing)=nothing,
  uptogradlogtarget::Union(Function, Nothing)=nothing,
  uptotensorlogtarget::Union(Function, Nothing)=nothing,
  uptodtensorlogtarget::Union(Function, Nothing)=nothing
)
  ContinuousMultivariateParameter(
    index,
    key,
    pdf,
    prior,
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )
end
