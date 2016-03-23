using Lora

data, header = dataset("swiss")

covariates = data[:, 1:end-1]
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome = data[:, end]

function ploglikelihood(p::Vector, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log(1+exp(Xp)))
end

function plogprior(p::Vector, v::Vector)
  -0.5*(dot(p, p)/v[1]+length(p)*log(6.283185307179586*v[1]))
end

λ = Hyperparameter(:λ)

X = Data(:X)

y = Data(:y)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

init = Any[(:p, v0[:p]), (:v, Any[v0[:λ], v0[:X], v0[:y], v0[:p]])]

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=4,
  autodiff=:reverse,
  init=init,
  order=2
)

model = likelihood_model([λ, X, y, p], isindexed=false)

sampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, outopts=outopts)

run(job)

chain = output(job)

mean(chain)