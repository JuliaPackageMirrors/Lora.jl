### ContinuousParameterIOStream

type ContinuousParameterIOStream <: ParameterIOStream
  value::Union(IOStream, Nothing)
  loglikelihood::Union(IOStream, Nothing)
  logprior::Union(IOStream, Nothing)
  logtarget::Union(IOStream, Nothing)
  gradloglikelihood::Union(IOStream, Nothing)
  gradlogprior::Union(IOStream, Nothing)
  gradlogtarget::Union(IOStream, Nothing)
  tensorloglikelihood::Union(IOStream, Nothing)
  tensorlogprior::Union(IOStream, Nothing)
  tensorlogtarget::Union(IOStream, Nothing)
  dtensorloglikelihood::Union(IOStream, Nothing)
  dtensorlogprior::Union(IOStream, Nothing)
  dtensorlogtarget::Union(IOStream, Nothing)
  diagnostics::Union(IOStream, Nothing)
  size::Tuple
  n::Int
  write::Function

  ContinuousParameterIOStream(streams::Vector{Union(IOStream, Nothing)}, size::Tuple, n::Int) = begin
    instance = new()

    for i in 1:14
      setfield!(instance, main_cpstate_fields[i], streams[i])
    end

    instance.size = size
    instance.n = n

    instance.write = eval(codegen_write_continuous_parameter_iostream(instance))

    instance
  end
end

ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  monitor::Vector{Bool},
  filepath::AbstractString,
  filesuffix::AbstractString
) =
  ContinuousParameterIOStream(
    [
      monitor[i] == false ? nothing : open(joinpath(filepath, string(main_cpstate_fields[i]), "."*filesuffix))
      for i in 1:14
    ],
    size,
    n
  )

ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  monitor::Vector{Symbol},
  filepath::AbstractString,
  filesuffix::AbstractString
) =
  ContinuousParameterIOStream(
    [
      main_cpstate_fields[i] in monitor ?
        open(joinpath(filepath, string(main_cpstate_fields[i]), "."*filesuffix)) :
        true
      for i in 1:14
    ],
    size,
    n
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

  if iostream.diagnostics != nothing
    push!(body, :(write($(iostream).diagnostics, join(values($(:_state).diagnostics), ','), "\n")))
  end

  @gensym write_continuous_parameter_iostream

  quote
    function $write_continuous_parameter_iostream(_state::ContinuousParameterState)
      $(body...)
    end
  end
end
