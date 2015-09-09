### VariableIOStreams

abstract VariableIOStream

### Variable IOStream subtypes

## GenericVariableIOStream

type GenericVariableIOStream <: VariableIOStream
  stream::IOStream
  size::Tuple
  n::Int
end

GenericVariableIOStream(filename::AbstractString, size::Tuple, n::Int) =
  GenericVariableIOStream(open(filename), size, n)

Base.close(iostream::GenericVariableIOStream) = close(iostream.stream)

Base.write(iostream::GenericVariableIOStream, state::GenericVariableState) =
  write(iostream.stream, join(state.value, ','), "\n")

Base.write(iostream::GenericVariableIOStream, nstate::GenericVariableNState) =
  write(iostream.stream, join(nstate.value, ','), "\n")

function Base.read{N<:Number}(iostream::GenericVariableIOStream, T::N)
  nstate::GenericVariableNState
  l = length(iostream.size)

  if l == 1
    if iostream.size[1] == 1
      nstate = UnivariateGenericVariableNState(vec(readdlm(iostream.stream, ',', T)), iostream.n)
    elseif iostream.size[1] > 1
      nstate = MultivariateGenericVariableNState(readdlm(iostream.stream, ',', T)', iostream.size[1], iostream.n)
    else
      error("GenericVariableIOStream size must be > 0, got $(iostream.size[1])")
    end
  elseif l > 1
    nstate = MatrixvariateGenericVariableNState(T, iostream.size, iostream.n)
    statelen = (nstate.size)^3
    line = 1
    while !eof(iostream.stream)
      nstate.value[1+(line-1)*statelen:line*statelen] =
        T[parse(T, c) for c in split(rstrip(readline(iostream.stream)), ',')]
      line += 1
    end
  else
    error("GenericVariableIOStream.size must be a tuple of length > 0, got $(iostream.size) length")
  end

  nstate
end

## ParameterIOStream

type ParameterIOStream <: VariableIOStream
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
  diagnostics::Union(Vector{IOStream}, Nothing)
  size::Tuple
  n::Int
  write::Function
end
