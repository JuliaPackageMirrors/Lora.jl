using Base.Test
using Distributions
using Lora

fields = {
  :pdf=>:pdf,
  :prior=>:prior,
  :spdf=>:setpdf,
  :sprior=>:setprior,
  :ll=>:loglikelihood!,
  :lp=>:logprior!,
  :lt=>:logtarget!,
  :gll=>:gradloglikelihood!,
  :glp=>:gradlogprior!,
  :glt=>:gradlogtarget!,
  :tll=>:tensorloglikelihood!,
  :tlp=>:tensorlogprior!,
  :tlt=>:tensorlogtarget!,
  :dtll=>:dtensorloglikelihood!,
  :dtlp=>:dtensorlogprior!,
  :dtlt=>:dtensorlogtarget!,
  :uptoglt=>:uptogradlogtarget!,
  :uptotlt=>:uptotensorlogtarget!,
  :uptodtlt=>:uptodtensorlogtarget!
}

println("    Testing ContinuousUnivariateParameter constructors:")

println("      Initialization via index and key fields...")

p = ContinuousUnivariateParameter(1, :p)

for field in values(fields)
  @test getfield(p, field) == nothing
end

println("      Initialization via pdf field...")

v = 5.18
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 6.11
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(1, :p, pdf=Normal(nstates[:μ].value))

distribution = Normal(μ)
p.pdf == distribution
lt, glt = logpdf(distribution, v), gradlogpdf(distribution, v)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

v = -11.87
pstate.value = v
μ = -20.2
nstates[:μ].value = μ

p.pdf = Normal(nstates[:μ].value)

distribution = Normal(μ)
p.pdf == distribution
lt, glt = logpdf(distribution, v), gradlogpdf(distribution, v)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via prior field...")

v = 1.25
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
σ = 10.
nstates[:σ] = UnivariateGenericVariableState(σ)

p = ContinuousUnivariateParameter(1, :p, prior=Normal(0., nstates[:σ].value))

distribution = Normal(0., σ)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, v)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, v)

for field in [
  :pdf, :spdf,
  :sprior,
  :ll, :lt,
  :gll, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

v = -0.21
pstate.value = v
σ = 1.
nstates[:σ].value = σ

p.prior = Normal(0., nstates[:σ].value)

distribution = Normal(0., σ)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, v)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, v)

for field in [
  :pdf, :spdf,
  :sprior,
  :ll, :lt,
  :gll, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setpdf field...")

v = 3.79
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 5.4
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(1, :p, setpdf=(pstates, nstates) -> Normal(nstates[:μ].value))
p.setpdf(pstate, nstates)

distribution = Normal(μ)
p.pdf == distribution
lt, glt = logpdf(distribution, v), gradlogpdf(distribution, v)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

v = -1.91
pstate.value = v
μ = 0.12
nstates[:μ].value = μ

p.setpdf(pstate, nstates)

distribution = Normal(μ)
p.pdf == distribution
lt, glt = logpdf(distribution, v), gradlogpdf(distribution, v)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setprior field...")

v = 3.55
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
σ = 2.
nstates[:σ] = UnivariateGenericVariableState(σ)

p = ContinuousUnivariateParameter(1, :p, setprior=(pstates, nstates) -> Normal(0., nstates[:σ].value))
p.setprior(pstate, nstates)

distribution = Normal(0., σ)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, v)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, v)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

v = -2.67
pstate.value = v
σ = 5.
nstates[:σ].value = σ

p.setprior(pstate, nstates)

distribution = Normal(0., σ)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, v)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, v)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal conjugacy: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood! and logprior! fields...")

v = -2.637
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = -1.88
nstates[:μ] = UnivariateGenericVariableState(μ)
σ = 1.
nstates[:σ] = UnivariateGenericVariableState(σ)
μ0 = 0.
nstates[:μ0] = UnivariateGenericVariableState(μ0)
σ0 = 1.
nstates[:σ0] = UnivariateGenericVariableState(σ0)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*((pstate.value-nstates[:μ].value)^2/(nstates[:σ].value^2)+log(2*pi))-log(nstates[:σ].value)

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*((nstates[:μ].value-nstates[:μ0].value)^2/(nstates[:σ0].value^2)+log(2*pi))-log(nstates[:σ0].value)

p = ContinuousUnivariateParameter(1, :p, loglikelihood=llf, logprior=lpf)

ld = Normal(μ, σ)
pd = Normal(μ0, σ0)
ll, lp = logpdf(ld, v), logpdf(pd, μ)
lt = ll+lp
p.loglikelihood!(pstate, nstates)
@test pstate.loglikelihood == ll
p.logprior!(pstate, nstates)
@test pstate.logprior == lp

pstate = ContinuousUnivariateParameterState(v)

p.logtarget!(pstate, nstates)
@test (pstate.loglikelihood, pstate.logprior, pstate.logtarget) == (ll, lp, lt)

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :gll, :glp, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! field...")

v = -1.28
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 9.4
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)^2
)

p.logtarget!(pstate, nstates)
@test 0.5*(pstate.logtarget-log(2*pi)) == logpdf(Normal(μ), v)

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :ll, :lp,
  :gll, :glp, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal conjugacy: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood!, logprior!, gradloglikelihood! and gradlogprior! fields...")

v = 6.69
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 5.43
nstates[:μ] = UnivariateGenericVariableState(μ)
σ = 1.
nstates[:σ] = UnivariateGenericVariableState(σ)
μ0 = 0.
nstates[:μ0] = UnivariateGenericVariableState(μ0)
σ0 = 1.
nstates[:σ0] = UnivariateGenericVariableState(σ0)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*((pstate.value-nstates[:μ].value)^2/(nstates[:σ].value^2)+log(2*pi))-log(nstates[:σ].value)

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*((nstates[:μ].value-nstates[:μ0].value)^2/(nstates[:σ0].value^2)+log(2*pi))-log(nstates[:σ0].value)

gllf(pstate, nstates) = pstate.gradloglikelihood = (pstate.value-nstates[:μ].value)/(nstates[:σ].value^2)

glpf(pstate, nstates) =
  pstate.gradlogprior = -(nstates[:μ].value-nstates[:μ0].value)/(nstates[:σ0].value^2)

p = ContinuousUnivariateParameter(1, :p, loglikelihood=llf, logprior=lpf, gradloglikelihood=gllf, gradlogprior=glpf)

ld = Normal(μ, σ)
pd = Normal(μ0, σ0)
ll, lp = logpdf(ld, v), logpdf(pd, μ)
lt = ll+lp
gll, glp = -gradlogpdf(ld, v), gradlogpdf(pd, μ)
glt = gll+glp
p.loglikelihood!(pstate, nstates)
@test pstate.loglikelihood == ll
p.logprior!(pstate, nstates)
@test pstate.logprior == lp
p.gradloglikelihood!(pstate, nstates)
@test pstate.gradloglikelihood == gll
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == glp

pstate = ContinuousUnivariateParameterState(v)

p.logtarget!(pstate, nstates)
@test (pstate.loglikelihood, pstate.logprior, pstate.logtarget) == (ll, lp, lt)
p.gradlogtarget!(pstate, nstates)
@test (pstate.gradloglikelihood, pstate.gradlogprior, pstate.gradlogtarget) == (gll, glp, glt)

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.loglikelihood, pstate.logprior, pstate.logtarget) == (ll, lp, lt)
@test (pstate.gradloglikelihood, pstate.gradlogprior, pstate.gradlogtarget) == (gll, glp, glt)

for field in [:pdf, :prior, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! and gradlogtarget! fields...")

v = -4.29
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 2.2
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)^2,
  gradlogtarget=(pstate, nstates) -> pstate.gradlogtarget = -2*(pstate.value-nstates[:μ].value)
)

distribution = Normal(μ)
lt, glt = logpdf(distribution, v), gradlogpdf(distribution, v)
p.logtarget!(pstate, nstates)
@test 0.5*(pstate.logtarget-log(2*pi)) == lt
p.gradlogtarget!(pstate, nstates)
@test 0.5*pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(v)

p.uptogradlogtarget!(pstate, nstates)
@test (0.5*(pstate.logtarget-log(2*pi)), 0.5*pstate.gradlogtarget) == (lt, glt)

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :ll, :lp,
  :gll, :glp,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end
