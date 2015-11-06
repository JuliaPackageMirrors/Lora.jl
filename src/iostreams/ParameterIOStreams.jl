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
  diagnosticvalues::Union{IOStream, Void}
  names::Vector{AbstractString}
  size::Tuple
  n::Int
  diagnostickeys::Vector{Symbol}
  write::Function

  function ContinuousParameterIOStream(
    size::Tuple,
    n::Int,
    streams::Vector{Union{IOStream, Void}},
    diagnostickeys::Vector{Symbol}=Symbol[],
    filenames::Vector{AbstractString}=[(streams[i] == nothing) ? "" : streams[i].name[7:end-1] for i in 1:14]
  )
    instance = new()

    fnames = fieldnames(ContinuousParameterIOStream)
    for i in 1:14
      setfield!(instance, fnames[i], streams[i])
    end

    instance.names = filenames

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.write = eval(codegen_write_continuous_parameter_iostream(instance))

    instance
  end
end

function ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  filenames::Vector{AbstractString},
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(ContinuousParameterIOStream)
  ContinuousParameterIOStream(
    size,
    n,
    [isempty(filenames[i]) ? nothing : open(filenames[i], mode) for i in 1:14],
    diagnostickeys,
    filenames
  )
end

function ContinuousParameterIOStream(
  size::Tuple,
  n::Int;
  monitor::Vector{Bool}=[true; fill(false, 13)],
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(ContinuousParameterIOStream)
  filenames = AbstractString[monitor[i] == false ? "" : joinpath(filepath, string(fnames[i])*"."*filesuffix) for i in 1:14]

  ContinuousParameterIOStream(size, n, filenames, diagnostickeys, mode)
end

function ContinuousParameterIOStream(
  size::Tuple,
  n::Int,
  monitor::Vector{Symbol};
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(ContinuousParameterIOStream)
  ContinuousParameterIOStream(
    size,
    n,
    monitor=[fnames[i] in monitor ? true : false for i in 1:14],
    filepath=filepath,
    filesuffix=filesuffix,
    diagnostickeys=diagnostickeys,
    mode=mode
  )
end

# To visually inspect code generation via codegen_write_continuous_parameter_iostream, try for example
# using Lora
#
# iostream = ContinuousParameterIOStream("w", (), 4, filepath="")
# Lora.codegen_write_continuous_parameter_iostream(iostream)
# close(iostream)

function codegen_write_continuous_parameter_iostream(iostream::ContinuousParameterIOStream)
  body = []
  fnames = fieldnames(ContinuousParameterIOStream)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  for i in 1:13
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(
        body,
        :(write(getfield($(iostream), $(QuoteNode(f))), join(getfield($(:_state), $(QuoteNode(f))), ','), "\n"))
      )
    end
  end

  if iostream.diagnosticvalues != nothing
    push!(body, :(write($(iostream).diagnosticvalues, join($(:_state).diagnosticvalues, ','), "\n")))
  end

  @gensym write_continuous_parameter_iostream

  quote
    function $write_continuous_parameter_iostream(_state::ContinuousParameterState)
      $(body...)
    end
  end
end

function Base.flush(iostream::ContinuousParameterIOStream)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      flush(iostream.(fnames[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    flush(iostream.diagnosticvalues)
  end
end

function Base.close(iostream::ContinuousParameterIOStream)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      close(iostream.(fnames[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    close(iostream.diagnosticvalues)
  end
end

function Base.open(iostream::ContinuousParameterIOStream, mode::AbstractString="w")
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      iostream.(fnames[i]) = open(iostream.names[i], mode)
    end
  end

  if iostream.diagnosticvalues != nothing
    iostream.diagnosticvalues = open(iostream.names[14], mode)
  end
end

function Base.mark(iostream::ContinuousParameterIOStream)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      mark(iostream.(fnames[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    mark(iostream.diagnosticvalues)
  end
end

function Base.reset(iostream::ContinuousParameterIOStream)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      reset(iostream.(fnames[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    reset(iostream.diagnosticvalues)
  end
end

function Base.write(iostream::ContinuousParameterIOStream, nstate::ContinuousUnivariateParameterNState)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end

  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function Base.write(iostream::ContinuousParameterIOStream, nstate::ContinuousMultivariateParameterNState)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end
  for i in (1, 5, 6, 7)
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i])', ',')
    end
  end
  for i in 8:10
    if iostream.(fnames[i]) != nothing
      statelen = abs2(iostream.size)
      for i in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
      end
    end
  end
  for i in 11:13
    if iostream.(fnames[i]) != nothing
      statelen = iostream.size^3
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
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', N)))
    end
  end

  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function Base.read!{N<:AbstractFloat}(
  iostream::ContinuousParameterIOStream,
  nstate::ContinuousMultivariateParameterNState{N}
)
  fnames = fieldnames(ContinuousParameterIOStream)
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', N)))
    end
  end
  for i in (1, 5, 6, 7)
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], readdlm(iostream.(fnames[i]), ',', N)')
    end
  end
  for i in 8:10
    if iostream.(fnames[i]) != nothing
      statelen = abs2(iostream.size)
      line = 1
      while !eof(iostream.stream)
        nstate.value[1+(line-1)*statelen:line*statelen] =
          [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
        line += 1
      end
    end
  end
  for i in 11:13
    if iostream.(fnames[i]) != nothing
      statelen = iostream.size^3
      line = 1
      while !eof(iostream.stream)
        nstate.value[1+(line-1)*statelen:line*statelen] =
          [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
        line += 1
      end
    end
  end

  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function Base.read{N<:AbstractFloat}(iostream::ContinuousParameterIOStream, T::Type{N})
  nstate::ContinuousParameterNState
  fnames = fieldnames(ContinuousParameterIOStream)
  l = length(iostream.size)

  if l == 0
    nstate = ContinuousUnivariateParameterNState(
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  elseif l == 1
    nstate = ContinuousMultivariateParameterNState(
      iostream.size[1],
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  else
    error("BasicVariableIOStream.size must be a tuple of length 0 or 1, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end
