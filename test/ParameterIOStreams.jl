using Base.Test
using Lora

filepath = dirname(@__FILE__)
filesuffix = "csv"
filenames = Array(AbstractString, 14)
for i in 1:13
  filenames[i] = joinpath(filepath, string(Lora.main_cpstate_fields[i])*"."*filesuffix)
end
filenames[14] = joinpath(filepath, "diagnostics"*"."*filesuffix)

println("    Testing ContinuousParameterIOStream constructors and close method...")

iostreamsize = ()
iostreamn = 10
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

iostreamsize = (3,)
iostreamn = 5
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

iostreamsize = (2,)
iostreamn = 7
iostream = ContinuousParameterIOStream(
  "w", iostreamsize, iostreamn, [:value, :logtarget, :diagnostics], diagnostickeys=[:accept], filepath=filepath
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

println("      Interaction with ContinuousUnivariateMCChain...")

println("      Interaction with ContinuousMultivariateParameterState...")

println("      Interaction with ContinuousMultivariateMCChain...")
