using Base.Test
using Lora

filename = joinpath(dirname(@__FILE__), "sample.txt")
# filename = "sample.txt"

println("    Testing GenericVariableIOStream constructors and methods...")

println("      Interaction with UnivariateGenericVariableState...")

nstatev = Float64[1.87, -4.5, 29.55, -0.91, 0.16]
iostreamsize = (1,)
iostreamn = length(nstatev)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
for v in nstatev
  write(iostream, UnivariateGenericVariableState(v))
end
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstate = read(iostream, Float64)
close(iostream)

rm(filename)

@test isa(nstate, UnivariateGenericVariableNState{Float64})
@test nstate.value == nstatev
@test nstate.n == iostream.n

println("      Interaction with UnivariateGenericVariableNState...")

nstatev = Float32[11.5, -41.22, -5.62, 1.98, 7.16]
iostreamsize = (1,)
iostreamn = length(nstatev)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
nstatein = UnivariateGenericVariableNState(nstatev)
write(iostream, nstatein)
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstateout = read(iostream, Float32)
close(iostream)

rm(filename)

@test isa(nstateout, UnivariateGenericVariableNState{Float32})
@test nstateout.value == nstatein.value
@test nstateout.n == nstatein.n

println("      Interaction with MultivariateGenericVariableState...")

nstatev = Float64[8.11 -0.99 -4.19 0.1; 0.01 -0.02 1.4 8.47]
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
for i in 1:iostreamn
  write(iostream, MultivariateGenericVariableState(nstatev[:, i]))
end
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstate = read(iostream, Float64)
close(iostream)

rm(filename)

@test isa(nstate, MultivariateGenericVariableNState{Float64})
@test nstate.value == nstatev
@test (nstate.size,) == iostream.size
@test nstate.n == iostream.n

println("      Interaction with MultivariateGenericVariableNState...")

nstatev = Float32[-7.1 -1.19 -7.76 6.1; -3.8 4.2 3.7 2.21]
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
nstatein = MultivariateGenericVariableNState(nstatev)
write(iostream, nstatein)
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstateout = read(iostream, Float32)
close(iostream)

rm(filename)

@test isa(nstateout, MultivariateGenericVariableNState{Float32})
@test nstateout.value == nstatein.value
@test nstateout.size == nstatein.size
@test nstateout.n == nstatein.n
