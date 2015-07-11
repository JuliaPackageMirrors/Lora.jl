using Graphs, Lora

m = GenericModel(Variable[], Dependence[])

θ = ContinuousUnivariateParameter(1, :θ)
x = Data(2, :x)
λ = Hyperparameter(3, :λ)

add_vertex!(m, θ)
add_vertex!(m, x)
add_vertex!(m, λ)

add_edge!(m, x, θ)
add_edge!(m, λ, θ)

g = convert(GenericGraph, m)

topological_sort_by_dfs(g)
