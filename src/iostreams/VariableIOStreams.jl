### VariableIOStreams

abstract VariableIOStream

### Variable IOStream subtypes

## GenericVariableIOStream

type GenericVariableIOStream <: VariableIOStream
  stream::IOStream
  size::Tuple
  n::Int
end

GenericVariableIOStream(filename::AbstractString, mode::AbstractString, size::Tuple, n::Int) =
  GenericVariableIOStream(open(filename, mode), size, n)

Base.close(iostream::GenericVariableIOStream) = close(iostream.stream)

Base.write(iostream::GenericVariableIOStream, state::GenericVariableState) =
  write(iostream.stream, join(state.value, ','), "\n")

Base.write(iostream::GenericVariableIOStream, nstate::UnivariateGenericVariableNState) =
  writedlm(iostream.stream, nstate.value)

Base.write(iostream::GenericVariableIOStream, nstate::MultivariateGenericVariableNState) =
  writedlm(iostream.stream, nstate.value', ',')

function Base.write(iostream::GenericVariableIOStream, nstate::MatrixvariateGenericVariableNState)
  statelen = prod(nstate.size)
  for i in 1:nstate.n
    write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
  end
end

Base.read!{N<:Number}(iostream::GenericVariableIOStream, nstate::UnivariateGenericVariableNState{N}) =
  nstate.value = vec(readdlm(iostream.stream, ',', N))

Base.read!{N<:Number}(iostream::GenericVariableIOStream, nstate::MultivariateGenericVariableNState{N}) =
  nstate.value = readdlm(iostream.stream, ',', N)'

function Base.read!{N<:Number}(iostream::GenericVariableIOStream, nstate::MatrixvariateGenericVariableNState{N})
  statelen = prod(iostream.size)
  line = 1
  while !eof(iostream.stream)
    nstate.value[1+(line-1)*statelen:line*statelen] =
      N[parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
    line += 1
  end
end

function Base.read{N<:Number}(iostream::GenericVariableIOStream, T::Type{N})
  nstate::GenericVariableNState
  l = length(iostream.size)

  if l == 1
    if iostream.size[1] == 1
      nstate = UnivariateGenericVariableNState(T, iostream.n)
    elseif iostream.size[1] > 1
      nstate = MultivariateGenericVariableNState(T, iostream.size[1], iostream.n)
    else
      error("GenericVariableIOStream size must be > 0, got $(iostream.size[1])")
    end
  elseif l > 1
    nstate = MatrixvariateGenericVariableNState(T, iostream.size, iostream.n)
  else
    error("GenericVariableIOStream.size must be a tuple of length > 0, got $(iostream.size) length")
  end

  read!(iostream, nstate)
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
