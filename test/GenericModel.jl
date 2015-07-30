using Graphs
using Lora

println("    Testing GenericModel constructors...")

println("      Initialization without input arguments...")

m = GenericModel()

θ = ContinuousUnivariateParameter(1, :θ)
x = Data(2, :x)
λ = Hyperparameter(3, :λ)

add_vertex!(m, θ)
add_vertex!(m, x)
add_vertex!(m, λ)

add_edge!(m, x, θ)
add_edge!(m, λ, θ)

println("      Initialization via matrix-based specification of dependencies...")

θ = ContinuousUnivariateParameter(1, :θ)
x = Data(2, :x)
λ = Hyperparameter(3, :λ)

m = GenericModel([θ, x, λ], [x θ; λ θ])

println("    Testing conversion of GenericModel to GenericGraph...")

convert(GenericGraph, m)

println("    Testing topological sorting of GenericModel...")

topological_sort_by_dfs(m)

# println("    Testing plotting of GenericModel...")

# plot(g)
