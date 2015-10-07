using Base.Test
using Lora

filename = joinpath(dirname(@__FILE__), "sample.txt")
# filename = "sample.txt"

println("    Testing GenericVariableIOStream constructors and methods...")

println("      Interaction with UnivariateGenericVariableState...")

nstatev = Float64[1.87, -4.5, 29.55, -0.91, 0.16]
iostreamsize = ()
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
iostreamsize = ()
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

println("      Interaction with MatrixvariateGenericVariableState...")

nstatev = Array(Float64, 3, 4, 2)
nstatev[:, :, 1] = [
  0.765895  0.965563  0.777571  0.508304;
  0.782592  0.366614  0.736418  0.863816;
  0.691046  0.524103  0.922673  0.223199
]
nstatev[:, :, 2] = [
  0.304872  0.210775  0.757637  0.657834;
  0.46678   0.995023  0.565492  0.007096;
  0.50774   0.362511  0.616699  0.675748
]
iostreamsize = (size(nstatev, 1), size(nstatev, 2))
iostreamn = size(nstatev, 3)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
for i in 1:iostreamn
  write(iostream, MatrixvariateGenericVariableState(nstatev[:, :, i]))
end
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstate = read(iostream, Float64)
close(iostream)

rm(filename)

@test isa(nstate, MatrixvariateGenericVariableNState{Float64})
@test nstate.value == nstatev
@test nstate.size == iostream.size
@test nstate.n == iostream.n

println("      Interaction with MatrixvariateGenericVariableNState...")

nstatev = Array(Float32, 3, 4, 2)
nstatev[:, :, 1] = [
  0.411455  0.268814  0.793351  0.502612;
  0.302095  0.702721  0.412092  0.488215;
  0.200365  0.862643  0.263403  0.442472
]
nstatev[:, :, 2] = [
 0.183705  0.0208819  0.423192  0.90061;
 0.31084   0.115514   0.464941  0.488014;
 0.913261  0.559753   0.969542  0.931011
]
iostreamsize = (size(nstatev, 1), size(nstatev, 2))
iostreamn = size(nstatev, 3)

iostream = GenericVariableIOStream(filename, "w", iostreamsize, iostreamn)
nstatein = MatrixvariateGenericVariableNState(nstatev)
write(iostream, nstatein)
close(iostream)

iostream = GenericVariableIOStream(filename, "r", iostreamsize, iostreamn)
nstateout = read(iostream, Float32)
close(iostream)

rm(filename)

@test isa(nstateout, MatrixvariateGenericVariableNState{Float32})
@test nstateout.value == nstatein.value
@test nstateout.size == nstatein.size
@test nstateout.n == nstatein.n
