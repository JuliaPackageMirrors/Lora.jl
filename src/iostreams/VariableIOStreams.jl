### VariableIOStreams

abstract VariableIOStream

### Variable IOStream subtypes

## GenericVariableIOStream

type GenericVariableIOStream <: VariableIOStream
  stream::IOStream
  size::Tuple
  n::Int
end

GenericVariableIOStream(filename::String, size::Tuple, n::Int) = GenericVariableIOStream(open(filename), size, n)

Base.write(iostream::GenericVariableIOStream, nstate::GenericVariableNState) =
  write(iostream.stream, join(nstate.value, ","), "\n")

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
