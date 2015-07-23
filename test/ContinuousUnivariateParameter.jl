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

println("      Testing univariate parameter initialization by setting only its index and key fields...")

p = ContinuousUnivariateParameter(1, :p)

for field in values(fields)
  @test getfield(p, field) == nothing
end

println("      Testing univariate parameter initialization via its pdf field...")

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

println("      Testing univariate parameter initialization via its prior field...")

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

println("      Testing univariate parameter initialization via its setpdf field...")

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

println("      Testing univariate parameter initialization via its setprior field...")

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

println("      Testing univariate parameter initialization via its logtarget field (unnormalized normal target)...")

v = -1.28
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 9.4
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)⋅(pstate.value-nstates[:μ].value)
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

println(
  "      Testing univariate parameter initialization via its logtarget and gradlogtarget fields"*
  "(unnormalized normal target)..."
)

v = -4.29
pstate = ContinuousUnivariateParameterState(v)
nstates = Dict{Symbol, VariableState}()
μ = 2.2
nstates[:μ] = UnivariateGenericVariableState(μ)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)⋅(pstate.value-nstates[:μ].value),
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
