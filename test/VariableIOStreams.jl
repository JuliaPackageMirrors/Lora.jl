using Base.Test
using Lora

filename = joinpath(dirname(@__FILE__), "sample.txt")
# filename = "sample.txt"

println("    Testing GenericVariableIOStream constructors and methods...")

println("      Interaction with UnivariateGenericVariableState...")

iostream = GenericVariableIOStream(filename, "w", (1,), 5)

sv = Float64[1.87, -4.5, 29.55, -0.91, 0.16]

for v in sv
  write(iostream, UnivariateGenericVariableState(v))
end

close(iostream)

iostream = GenericVariableIOStream(filename, "r", (1,), 5)

nstate = read(iostream, Float64)

@test isa(nstate, UnivariateGenericVariableNState{Float64})
@test nstate.value == sv
@test nstate.n == iostream.n

close(iostream)

rm(filename)

println("      Interaction with UnivariateGenericVariableNState...")

iostream = GenericVariableIOStream(filename, "w", (1,), 5)

sv = Float32[11.5, -41.22, -5.62, 1.98, 7.16]

nstatein = UnivariateGenericVariableNState(sv)

write(iostream, nstatein)

close(iostream)

iostream = GenericVariableIOStream(filename, "r", (1,), 5)

nstateout = read(iostream, Float32)

@test isa(nstateout, UnivariateGenericVariableNState{Float32})
@test nstateout.value == nstatein.value
@test nstateout.n == nstatein.n

close(iostream)

rm(filename)
