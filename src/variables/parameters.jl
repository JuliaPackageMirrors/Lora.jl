### Astract Parameter types

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

abstract ContinuousParameter{S<:ValueSupport, F<:VariateForm} <: Parameter{S, F}

### Parameter subtypes

## ContinuousUnivariateParameter

# Guidelines for usage of inner constructors of continuous parameter types:
# 1) Function fields have higher priority than implicitly derived definitions via the pdf field
# 2) Target-related fields have higher priority than implicitly derived likelihood+prior fields
# 3) Upto-related fields have higher priority than implicitly derived Function tuples

type ContinuousUnivariateParameter <: ContinuousParameter{Continuous, Univariate}
  index::Int
  key::Symbol
  pdf::Union{ContinuousUnivariateDistribution, Void}
  prior::Union{ContinuousUnivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  gradloglikelihood!::Union{Function, Void}
  gradlogprior!::Union{Function, Void}
  gradlogtarget!::Union{Function, Void}
  tensorloglikelihood!::Union{Function, Void}
  tensorlogprior!::Union{Function, Void}
  tensorlogtarget!::Union{Function, Void}
  dtensorloglikelihood!::Union{Function, Void}
  dtensorlogprior!::Union{Function, Void}
  dtensorlogtarget!::Union{Function, Void}
  uptogradlogtarget!::Union{Function, Void}
  uptotensorlogtarget!::Union{Function, Void}
  uptodtensorlogtarget!::Union{Function, Void}

  ContinuousUnivariateParameter(
    index::Int,
    key::Symbol,
    pdf::Union{ContinuousUnivariateDistribution, Void},
    prior::Union{ContinuousUnivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    gll::Union{Function, Void},
    glp::Union{Function, Void},
    glt::Union{Function, Void},
    tll::Union{Function, Void},
    tlp::Union{Function, Void},
    tlt::Union{Function, Void},
    dtll::Union{Function, Void},
    dtlp::Union{Function, Void},
    dtlt::Union{Function, Void},
    uptoglt::Union{Function, Void},
    uptotlt::Union{Function, Void},
    uptodtlt::Union{Function, Void}
  ) = begin
    instance = new()
    instance.index = index
    instance.key = key
    instance.pdf = pdf
    instance.prior = prior

    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
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
      if isa(args[i], Function) && 
        isgeneric(args[i]) &&
        !method_exists(args[i], (ContinuousUnivariateParameterState, Vector{VariableState}))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          (state::ContinuousUnivariateParameterState, states::Vector{VariableState}) -> 
          setfield!(instance, distribution, args[i](state, states))
        else
          args[i]
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = args[3]
    instance.gradloglikelihood! = args[6]

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf),
      (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if args[i] == nothing && (
          (isa(prior, ContinuousUnivariateDistribution) && method_exists(f, (typeof(prior), eltype(prior)))) ||
          isa(args[2], Function)
        )
          (state::ContinuousUnivariateParameterState, states::Vector{VariableState}) ->
          setfield!(state, spfield, f(instance.prior, state.value))
        else
          args[i]
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
        if args[i] == nothing
          if isa(args[i-2], Function) && isa(getfield(instance, ppfield), Function)
            function (state::ContinuousUnivariateParameterState, states::Vector{VariableState})
              getfield(instance, plfield)(state, states)
              getfield(instance, ppfield)(state, states)
              setfield!(state, stfield, getfield(state, slfield)+getfield(state, spfield))
            end
          elseif (isa(pdf, ContinuousUnivariateDistribution) && method_exists(f, (typeof(pdf), eltype(pdf)))) ||
            isa(args[1], Function)
            (state::ContinuousUnivariateParameterState, states::Vector{VariableState}) ->
            setfield!(state, stfield, f(instance.pdf, state.value))
          end
        else
          args[i]
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = args[9]
    instance.dtensorloglikelihood! = args[12]

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = args[10]
    instance.dtensorlogprior! = args[13]

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
        if args[i] == nothing && isa(args[i-2], Function) && isa(args[i-1], Function)
          function (state::ContinuousUnivariateParameterState, states::Vector{VariableState})
            getfield(instance, plfield)(state, states)
            getfield(instance, ppfield)(state, states)
            setfield!(state, stfield, getfield(state, slfield)+getfield(state, spfield))
          end
        else
          args[i]
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if args[15] == nothing && isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
        function (state::ContinuousUnivariateParameterState, states::Vector{VariableState})
          instance.logtarget!(state, states)
          instance.gradlogtarget!(state, states)
        end
      else
        args[15]
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if args[16] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function)
        function (state::ContinuousUnivariateParameterState, states::Vector{VariableState})
          instance.logtarget!(state, states)
          instance.gradlogtarget!(state, states)
          instance.tensorlogtarget!(state, states)
        end
      else
        args[16]
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if args[17] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function) &&
        isa(instance.dtensorlogtarget!, Function)
        function (state::ContinuousUnivariateParameterState, states::Vector{VariableState})
          instance.logtarget!(state, states)
          instance.gradlogtarget!(state, states)
          instance.tensorlogtarget!(state, states)
          instance.dtensorlogtarget!(state, states)
        end
      else
        args[17]
      end
    )

    instance
  end
end

function ContinuousUnivariateParameter(
  index::Int,
  key::Symbol;
  pdf::Union{ContinuousUnivariateDistribution, Void}=nothing,
  prior::Union{ContinuousUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing
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

type ContinuousMultivariateParameter <: ContinuousParameter{Continuous, Multivariate}
  index::Int
  key::Symbol
  pdf::Union{ContinuousMultivariateDistribution, Void}
  prior::Union{ContinuousMultivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  gradloglikelihood!::Union{Function, Void}
  gradlogprior!::Union{Function, Void}
  gradlogtarget!::Union{Function, Void}
  tensorloglikelihood!::Union{Function, Void}
  tensorlogprior!::Union{Function, Void}
  tensorlogtarget!::Union{Function, Void}
  dtensorloglikelihood!::Union{Function, Void}
  dtensorlogprior!::Union{Function, Void}
  dtensorlogtarget!::Union{Function, Void}
  uptogradlogtarget!::Union{Function, Void}
  uptotensorlogtarget!::Union{Function, Void}
  uptodtensorlogtarget!::Union{Function, Void}

  ContinuousMultivariateParameter(
    index::Int,
    key::Symbol,
    pdf::Union{ContinuousMultivariateDistribution, Void},
    prior::Union{ContinuousMultivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    gll::Union{Function, Void},
    glp::Union{Function, Void},
    glt::Union{Function, Void},
    tll::Union{Function, Void},
    tlp::Union{Function, Void},
    tlt::Union{Function, Void},
    dtll::Union{Function, Void},
    dtlp::Union{Function, Void},
    dtlt::Union{Function, Void},
    uptoglt::Union{Function, Void},
    uptotlt::Union{Function, Void},
    uptodtlt::Union{Function, Void}
  ) = begin
    instance = new()
    instance.index = index
    instance.key = key
    instance.pdf = pdf
    instance.prior = prior

    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
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
      if isa(args[i], Function) && isgeneric(args[i]) && !method_exists(args[i], (Vector{VariableState}, Int))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          (states::Vector{VariableState}, j::Int) -> setfield!(instance, distribution, args[i](states, j))
        else
          args[i]
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = args[3]
    instance.gradloglikelihood! = args[6]

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf),
      (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if args[i] == nothing && (
          (
            isa(prior, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(prior), Vector{eltype(prior)}))
          ) ||
          isa(args[2], Function)
        )
          (states::Vector{VariableState}, j::Int) -> setfield!(states[j], spfield, f(instance.prior, states[j].value))
        else
          args[i]
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
        if args[i] == nothing
          if isa(args[i-2], Function) && isa(getfield(instance, ppfield), Function)
            function (states::Vector{VariableState}, j::Int)
              getfield(instance, plfield)(states, j)
              getfield(instance, ppfield)(states, j)
              setfield!(states[j], stfield, getfield(states[j], slfield)+getfield(states[j], spfield))
            end
          elseif (
              isa(pdf, ContinuousMultivariateDistribution) &&
              method_exists(f, (typeof(pdf), Vector{eltype(pdf)}))
            ) ||
            isa(args[1], Function)
            (states::Vector{VariableState}, j::Int) -> setfield!(states[j], stfield, f(instance.pdf, states[j].value))
          end
        else
          args[i]
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = args[9]
    instance.dtensorloglikelihood! = args[12]

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = args[10]
    instance.dtensorlogprior! = args[13]

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
        if args[i] == nothing && isa(args[i-2], Function) && isa(args[i-1], Function)
          function (states::Vector{VariableState}, j::Int)
            getfield(instance, plfield)(states, j)
            getfield(instance, ppfield)(states, j)
            setfield!(states[j], stfield, getfield(states[j], slfield)+getfield(states[j], spfield))
          end
        else
          args[i]
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if args[15] == nothing && isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
        function (states::Vector{VariableState}, j::Int)
          instance.logtarget!(states, j)
          instance.gradlogtarget!(states, j)
        end
      else
        args[15]
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if args[16] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function)
        function (states::Vector{VariableState}, j::Int)
          instance.logtarget!(states, j)
          instance.gradlogtarget!(states, j)
          instance.tensorlogtarget!(states, j)
        end
      else
        args[16]
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if args[17] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function) &&
        isa(instance.dtensorlogtarget!, Function)
        function (states::Vector{VariableState}, j::Int)
          instance.logtarget!(states, j)
          instance.gradlogtarget!(states, j)
          instance.tensorlogtarget!(states, j)
          instance.dtensorlogtarget!(states, j)
        end
      else
        args[17]
      end
    )

    instance
  end
end

function ContinuousMultivariateParameter(
  index::Int,
  key::Symbol;
  pdf::Union{ContinuousMultivariateDistribution, Void}=nothing,
  prior::Union{ContinuousMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing
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
