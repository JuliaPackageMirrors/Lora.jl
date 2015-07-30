using Lora

println("    Testing likelihood_model generator...")

θ = ContinuousUnivariateParameter(1, :θ)
x = Data(2, :x)
λ = Hyperparameter(3, :λ)

likelihood_model([θ], data=[x], hyperparameters=[λ])

println("    Testing single_parameter_likelihood_model generator...")

θ = ContinuousUnivariateParameter(1, :θ)
x = Data(2, :x)
λ = Hyperparameter(3, :λ)

single_parameter_likelihood_model(θ, data=[x], hyperparameters=[λ])
