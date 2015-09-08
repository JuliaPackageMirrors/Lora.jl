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

Base.write(iostream::GenericVariableIOStream, state::GenericVariableState) =
  write(iostream.stream, join(state.value, ","), "\n")

Base.write(iostream::GenericVariableIOStream, nstate::GenericVariableNState) =
  write(iostream.stream, join(nstate.value, ","), "\n")

# function Base.read{N<:FloatingPoint}(iostream::GenericVariableIOStream, t::N)
#   nstate::GenericVariableNState

#   if length(iostream.size) == 1
#     if iostream.size[1] == 1
#       nstate = UnivariateGenericVariableNState(N, s)
#     else
#     end
#   else
#   end

#   nstate
# end

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
