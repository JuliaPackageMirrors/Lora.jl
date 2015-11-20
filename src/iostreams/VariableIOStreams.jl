### Abstract variable IOStreams

abstract VariableIOStream

### BasicVariableIOStream

type BasicVariableIOStream <: VariableIOStream
  stream::IOStream
  size::Tuple
  n::Int
end

BasicVariableIOStream(filename::AbstractString, size::Tuple, n::Int, mode::AbstractString="w") =
  BasicVariableIOStream(open(filename, mode), size, n)

Base.close(iostream::BasicVariableIOStream) = close(iostream.stream)

Base.write(iostream::BasicVariableIOStream, state::BasicVariableState) = write(iostream.stream, join(state.value, ','), "\n")

Base.write(iostream::BasicVariableIOStream, nstate::UnivariateBasicVariableNState) = writedlm(iostream.stream, nstate.value)

Base.write(iostream::BasicVariableIOStream, nstate::MultivariateBasicVariableNState) =
  writedlm(iostream.stream, nstate.value', ',')

function Base.write(iostream::BasicVariableIOStream, nstate::MatrixvariateBasicVariableNState)
  statelen = prod(nstate.size)
  for i in 1:nstate.n
    write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
  end
end

Base.read!{N<:Number}(iostream::BasicVariableIOStream, nstate::UnivariateBasicVariableNState{N}) =
  nstate.value = vec(readdlm(iostream.stream, ',', N))

Base.read!{N<:Number}(iostream::BasicVariableIOStream, nstate::MultivariateBasicVariableNState{N}) =
  nstate.value = readdlm(iostream.stream, ',', N)'

function Base.read!{N<:Number}(iostream::BasicVariableIOStream, nstate::MatrixvariateBasicVariableNState{N})
  statelen = prod(iostream.size)
  line = 1
  while !eof(iostream.stream)
    nstate.value[1+(line-1)*statelen:line*statelen] = [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
    line += 1
  end
end

function Base.read{N<:Number}(iostream::BasicVariableIOStream, T::Type{N})
  nstate::BasicVariableNState
  l = length(iostream.size)

  if l == 0
    nstate = UnivariateBasicVariableNState(iostream.n, T)
  elseif l == 1
    nstate = MultivariateBasicVariableNState(iostream.size[1], iostream.n, T)
  elseif l == 2
    nstate = MatrixvariateBasicVariableNState(iostream.size, iostream.n, T)
  else
    error("BasicVariableIOStream.size must be a tuple of length 0 or 1 or 2, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end
