using Base.Test
using Graphs
using Lora

println("    Testing conversion of variables to key vertices compatible with Graphs...")

θ = ContinuousUnivariateParameter(1, :θ)

gθ = convert(KeyVertex, θ)

@test θ.index == gθ.index
@test θ.key == gθ.key

x = Data(2, :x)

mvertices = [θ, x]
gvertices = convert(Vector{KeyVertex}, mvertices)

for i in 1:length(mvertices)
  @test mvertices[i].index == gvertices[i].index
  @test mvertices[i].key == gvertices[i].key
end
