using Base.Test
using Lora

filename = joinpath(dirname(@__FILE__), "sample.txt")
# filename = "sample.txt"

println("    Testing GenericParameterIOStream constructors and methods...")

println("      Interaction with ContinuousUnivariateParameterState...")
