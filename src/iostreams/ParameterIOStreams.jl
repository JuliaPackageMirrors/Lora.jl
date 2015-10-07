### Abstract parameter IOStreams

abstract ParameterIOStream <: VariableIOStream

### ContinuousParameterIOStream

type ContinuousParameterIOStream <: ParameterIOStream
  value::Union{IOStream, Void}
  loglikelihood::Union{IOStream, Void}
  logprior::Union{IOStream, Void}
  logtarget::Union{IOStream, Void}
  gradloglikelihood::Union{IOStream, Void}
  gradlogprior::Union{IOStream, Void}
  gradlogtarget::Union{IOStream, Void}
  tensorloglikelihood::Union{IOStream, Void}
  tensorlogprior::Union{IOStream, Void}
  tensorlogtarget::Union{IOStream, Void}
  dtensorloglikelihood::Union{IOStream, Void}
  dtensorlogprior::Union{IOStream, Void}
  dtensorlogtarget::Union{IOStream, Void}
  diagnostickeys::Vector{Symbol}
  diagnosticvalues::Union{IOStream, Void}
  size::Tuple
  n::Int
  write::Function

  ContinuousParameterIOStream(
    size::Tuple,
    n::Int,
    streams::Vector{Union{IOStream, Void}},
    diagnostickeys::Vector{Symbol}=Symbol[],
    diagnosticvalues::Union{IOStream, Void}=nothing
  ) = begin
    instance = new()

    for i in 1:13
      setfield!(instance, main_cpstate_fields[i], streams[i])
    end

    instance.diagnostickeys = diagnostickeys
    instance.diagnosticvalues = diagnosticvalues
    instance.size = size
    instance.n = n

    instance.write = eval(codegen_write_continuous_parameter_iostream(instance))

    instance
  end
end

ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  filepath::AbstractString="",
  filesuffix::AbstractString="csv"
) =
  ContinuousParameterIOStream(
    size,
    n,
    [
      monitor[i] == false ? nothing : open(joinpath(filepath, string(main_cpstate_fields[i])*"."*filesuffix))
      for i in 1:13
    ],
    diagnostickeys,
    isempty(diagnostickeys) ? nothing : open(joinpath(filepath, "diagnostics"*"."*filesuffix))
  )

ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  filepath::AbstractString="",
  filesuffix::AbstractString="csv"
) =
  ContinuousParameterIOStream(
    size,
    n,
    [
      main_cpstate_fields[i] in monitor ?
        open(joinpath(filepath, string(main_cpstate_fields[i]), "."*filesuffix)) :
        nothing
      for i in 1:13
    ],
    diagnostickeys,
    isempty(diagnostickeys) ? nothing : open(joinpath(filepath, "diagnostics"*"."*filesuffix))
  )

function codegen_write_continuous_parameter_iostream(iostream::ContinuousParameterIOStream)
  body = []

  for i in 1:13
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      push!(
        body,
        :(write($(iostream).(main_cpstate_fields[$i]), join($(:_state).(main_cpstate_fields[$i]), ','), "\n"))
      )
    end
  end

  if iostream.diagnosticvalues != nothing
    push!(body, :(write($(iostream).diagnosticvalues, join(values($(:_state).diagnosticvalues), ','), "\n")))
  end

  @gensym write_continuous_parameter_iostream

  quote
    function $write_continuous_parameter_iostream(_state::ContinuousParameterState)
      $(body...)
    end
  end
end

function Base.close(iostream::ContinuousParameterIOStream)
  for i in 1:13
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      close(getfield(iostream, main_cpstate_fields[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    close(iostream.diagnosticvalues)
  end
end

function Base.write(iostream::ContinuousParameterIOStream, nstate::ContinuousUnivariateParameterNState)
  for i in 1:13
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      writedlm(getfield(iostream, main_cpstate_fields[i]), getfield(nstate, main_cpstate_fields[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function Base.write(iostream::ContinuousParameterIOStream, nstate::ContinuousMultivariateParameterNState)
  for i in 2:4
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      writedlm(getfield(iostream, main_cpstate_fields[i]), getfield(nstate, main_cpstate_fields[i]))
    end
  end
  for i in (1, 5, 6, 7)
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      writedlm(getfield(iostream, main_cpstate_fields[i]), getfield(nstate, main_cpstate_fields[i])', ',')
    end
  end
  for i in 8:10
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      statelen = abs2(nstate.size)
      for i in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
      end
    end
  end
  for i in 11:13
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      statelen = nstate.size^3
      for i in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
      end
    end
  end

  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function Base.read!{N<:AbstractFloat}(
  iostream::ContinuousParameterIOStream,
  nstate::ContinuousUnivariateParameterNState{N}
)
  for i in 1:13
    if getfield(iostream, main_cpstate_fields[i]) != nothing
      setfield!(nstate, main_cpstate_fields[i], vec(readdlm(getfield(iostream, main_cpstate_fields[i]), ',', N)))
    end
  end

  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', N)'
  end
end
