using Base.Test
using Lora

# filepath = dirname(@__FILE__)
filepath = ""
filesuffix = "csv"
filenames = Array(AbstractString, 14)
for i in 1:13
  filenames[i] = joinpath(filepath, string(Lora.main_cpstate_fields[i])*"."*filesuffix)
end
filenames[14] = joinpath(filepath, "diagnostics"*"."*filesuffix)

println("    Testing ContinuousParameterIOStream constructors and close method...")

iostreamsize = ()
iostreamn = 4
iostream = ContinuousParameterIOStream("w", iostreamsize, iostreamn, filepath=filepath)

@test isa(iostream.value, IOStream)
for i in 2:13
  @test iostream.(Lora.main_cpstate_fields[i]) == nothing
end
@test length(iostream.diagnostickeys) == 0
@test iostream.diagnosticvalues == nothing
@test iostream.size == iostreamsize
@test iostream.n == iostreamn

close(iostream)
rm(filenames[1])

iostreamsize = (2,)
iostreamn = 4
iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn,
  monitor=[true; fill(false, 5); true; fill(false, 6); true], diagnostickeys=[:accept], filepath=filepath
)

@test isa(iostream.value, IOStream)
@test isa(iostream.gradlogtarget, IOStream)
for i in [2:6; 8:13]
  @test iostream.(Lora.main_cpstate_fields[i]) == nothing
end
@test length(iostream.diagnostickeys) == 1
@test iostream.size == iostreamsize
@test iostream.n == iostreamn

close(iostream)
for i in [1, 7, 14]; rm(filenames[i]); end

iostreamsize = (3,)
iostreamn = 7
iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn, [:value, :logtarget], diagnostickeys=[:accept], filepath=filepath
)

@test isa(iostream.value, IOStream)
@test isa(iostream.logtarget, IOStream)
for i in [2:3; 5:13]
  @test iostream.(Lora.main_cpstate_fields[i]) == nothing
end
@test length(iostream.diagnostickeys) == 1
@test iostream.size == iostreamsize
@test iostream.n == iostreamn

close(iostream)
for i in [1, 4, 14]; rm(filenames[i]); end

println("    Testing ContinuousParameterIOStream IO methods...")

println("      Interaction with ContinuousUnivariateParameterState...")

nstatev = Float64[5.70, 1.44, -1.21, 5.67]
iostreamsize = ()
iostreamn = length(nstatev)

iostream = ContinuousParameterIOStream("w", iostreamsize, iostreamn, filepath=filepath)
for v in nstatev
  iostream.write(ContinuousUnivariateParameterState(v))
end

close(iostream)

iostream = ContinuousParameterIOStream("r", iostreamsize, iostreamn, filepath=filepath)
nstate = read(iostream, Float64)

@test isa(nstate, ContinuousUnivariateMCChain{Float64})
@test nstate.value == nstatev
for i in 2:13
  @test length(nstate.(Lora.main_cpstate_fields[i])) == 0
end
@test length(nstate.diagnostickeys) == 0
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.n == iostream.n

close(iostream)
rm(filenames[1])

println("      Interaction with ContinuousUnivariateMCChain...")

nstatev = Float32[1.93, 98.46, -3.61, -0.99, 74.52, 9.90]
nstated = Any[false, true, true, false, true, false]'
iostreamsize = ()
iostreamn = length(nstatev)

iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn, [:value], diagnostickeys=[:accept], filepath=filepath
)
nstatein = ContinuousUnivariateMCChain(Float32, iostreamn)
nstatein.value = nstatev
nstatein.diagnosticvalues = nstated
write(iostream, nstatein)

close(iostream)

iostream = ContinuousParameterIOStream(
  "r", iostreamsize, iostreamn, [:value], diagnostickeys=[:accept], filepath=filepath
)
nstateout = read(iostream, Float32)

@test isa(nstateout, ContinuousUnivariateMCChain{Float32})
@test nstateout.value == nstatein.value
for i in 2:13
  @test length(nstateout.(Lora.main_cpstate_fields[i])) == 0
end
@test length(nstateout.diagnostickeys) == 1
@test nstateout.diagnosticvalues == nstatein.diagnosticvalues
@test nstateout.n == nstatein.n

close(iostream)
for i in [1, 14]; rm(filenames[i]); end

println("      Interaction with ContinuousMultivariateParameterState...")

nstatev = Float64[1.33 2.44 3.14 -0.82; 7.21 -9.75 -5.26 -0.63]
nstategll = Float64[3.13 -12.10 13.11 -0.99; 9.91 -5.25 -8.15 -9.69]
nstated = Any[false, true, true, false]'
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn, [:value, :gradloglikelihood], diagnostickeys=[:accept], filepath=filepath
)
for i in 1:iostreamn
  state = ContinuousMultivariateParameterState(nstatev[:, i], Symbol[], [:accept], [nstated[i]])
  state.gradloglikelihood = nstategll[:, i]
  iostream.write(state)
end

close(iostream)

iostream = ContinuousParameterIOStream(
  "r", iostreamsize, iostreamn, [:value, :gradloglikelihood], diagnostickeys=[:accept], filepath=filepath
)
nstate = read(iostream, Float64)

@test isa(nstate, ContinuousMultivariateMCChain{Float64})
@test nstate.value == nstatev
@test nstate.gradloglikelihood == nstategll
for i in [2:4; 6:13]
  @test length(nstate.(Lora.main_cpstate_fields[i])) == 0
end
@test length(nstate.diagnostickeys) == 1
@test nstate.diagnosticvalues == nstated
@test nstate.size == iostream.size[1]
@test nstate.n == iostream.n

close(iostream)
for i in [1, 5, 14]; rm(filenames[i]); end

println("      Interaction with ContinuousMultivariateMCChain...")

nstatev = Float32[-1.85 -0.09 0.36; -0.45 -0.85 1.91]
nstatell = Float32[-1.30, -1.65, -0.18]
nstatelt = Float32[-0.44, 0.72, -0.21]
nstated = Any[false true; true false; true true]'
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn, [:value, :loglikelihood, :logtarget], diagnostickeys=[:accept], filepath=filepath
)
nstatein = ContinuousMultivariateMCChain(
  Float32,
  iostreamsize[1],
  iostreamn,
  [:value, :loglikelihood, :logtarget],
  [:accept]
)
nstatein.value = nstatev
nstatein.loglikelihood = nstatell
nstatein.logtarget = nstatelt
nstatein.diagnosticvalues = nstated
write(iostream, nstatein)

close(iostream)

iostream = ContinuousParameterIOStream(
  "r", iostreamsize, iostreamn, [:value, :loglikelihood, :logtarget], diagnostickeys=[:accept], filepath=filepath
)
nstateout = read(iostream, Float32)

@test isa(nstateout, ContinuousMultivariateMCChain{Float32})
@test nstateout.value == nstatein.value
@test nstateout.loglikelihood == nstatein.loglikelihood
@test nstateout.logtarget == nstatein.logtarget
for i in [3; 5:13]
  @test length(nstateout.(Lora.main_cpstate_fields[i])) == 0
end
@test length(nstateout.diagnostickeys) == 1
@test nstateout.diagnosticvalues == nstatein.diagnosticvalues
@test nstateout.n == nstatein.n

close(iostream)
for i in [1, 2, 4, 14]; rm(filenames[i]); end
