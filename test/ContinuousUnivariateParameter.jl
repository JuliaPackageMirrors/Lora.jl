using Base.Test
using Distributions
using Lora

fields = Dict{Symbol, Symbol}(
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
)

println("    Testing ContinuousUnivariateParameter constructors:")

println("      Initialization via index and key fields...")

p = ContinuousUnivariateParameter(1, :p)

for field in values(fields)
  @test getfield(p, field) == nothing
end

println("      Initialization via pdf field...")

pv = 5.18
μv = 6.11
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(μv)]

p = ContinuousUnivariateParameter(1, :p, pdf=Normal(states[2].value))

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test states[1].logtarget == lt
p.gradlogtarget!(states, 1)
@test states[1].gradlogtarget == glt

states[1] = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -11.87
states[1].value = pv
μv = -20.2
states[2].value = μv

p.pdf = Normal(states[2].value)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test states[1].logtarget == lt
p.gradlogtarget!(states, 1)
@test states[1].gradlogtarget == glt

states[1] = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via prior field...")

pv = 1.25
σv = 10.
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(σv)]

p = ContinuousUnivariateParameter(1, :p, prior=Normal(0., states[2].value))

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(states, 1)
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states, 1)
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

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

pv = -0.21
states[1].value = pv
σv = 1.
states[2].value = σv

p.prior = Normal(0., states[2].value)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(states, 1)
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states, 1)
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

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

pv = 3.79
μv = 5.4
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(μv)]

p = ContinuousUnivariateParameter(1, :p, setpdf=(states, i) -> Normal(states[2].value))
p.setpdf(states, 1)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test states[1].logtarget == lt
p.gradlogtarget!(states, 1)
@test states[1].gradlogtarget == glt

states[1] = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -1.91
states[1].value = pv
μv = 0.12
states[2].value = μv

p.setpdf(states, 1)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test states[1].logtarget == lt
p.gradlogtarget!(states, 1)
@test states[1].gradlogtarget == glt

states[1] = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setprior field...")

pv = 3.55
σv = 2.
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(σv)]

p = ContinuousUnivariateParameter(1, :p, setprior=(states, i) -> Normal(0., states[2].value))
p.setprior(states, 1)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(states, 1)
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states, 1)
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -2.67
states[1].value = pv
σv = 5.
states[2].value = σv

p.setprior(states, 1)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(states, 1)
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states, 1)
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood! and logprior! fields...")

μv = -2.637
xv = -1.88
σv = 1.
μ0v = 0.
σ0v = 1.
states = VariableState[
  ContinuousUnivariateParameterState(μv),
  UnivariateGenericVariableState(xv),
  UnivariateGenericVariableState(σv),
  UnivariateGenericVariableState(μ0v),
  UnivariateGenericVariableState(σ0v)
]

llf(states, i) =
  states[i].loglikelihood =
  -0.5*((states[2].value-states[i].value)^2/(states[3].value^2)+log(2*pi))-log(states[3].value)

lpf(states, i) =
  states[i].logprior =
  -0.5*((states[i].value-states[4].value)^2/(states[5].value^2)+log(2*pi))-log(states[5].value)

μ = ContinuousUnivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
μ.loglikelihood!(states, 1)
@test_approx_eq states[1].loglikelihood ll
μ.logprior!(states, 1)
@test_approx_eq states[1].logprior lp

states[1] = ContinuousUnivariateParameterState(μv)

μ.logtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test_approx_eq states[1].logprior lp
@test_approx_eq states[1].logtarget lt

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :gll, :glp, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(μ, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! field...")

pv = -1.28
μv = 9.4
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(μv)]

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(states, i) -> states[i].logtarget = -(states[i].value-states[2].value)^2
)

p.logtarget!(states, 1)
@test_approx_eq 0.5*(states[1].logtarget-log(2*pi)) logpdf(Normal(μv), pv)

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

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood!, gradloglikelihood! and prior fields...")

μv = 5.59
xv = 4.11
σv = 1.
μ0v = 0.
σ0v = 1.
states = VariableState[
  ContinuousUnivariateParameterState(μv),
  UnivariateGenericVariableState(xv),
  UnivariateGenericVariableState(σv),
  UnivariateGenericVariableState(μ0v),
  UnivariateGenericVariableState(σ0v)
]

llf(states, i) =
  states[i].loglikelihood =
  -0.5*((states[2].value-states[i].value)^2/(states[3].value^2)+log(2*pi))-log(states[3].value)

gllf(states, i) = states[i].gradloglikelihood = (states[2].value-states[i].value)/(states[3].value^2)

μ = ContinuousUnivariateParameter(
  1,
  :μ,
  loglikelihood=llf,
  gradloglikelihood=gllf,
  prior=Normal(states[4].value, states[5].value)
)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states, 1)
@test_approx_eq states[1].loglikelihood ll
μ.logprior!(states, 1)
@test states[1].logprior == lp
μ.gradloglikelihood!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
μ.gradlogprior!(states, 1)
@test states[1].gradlogprior == glp

pstate = ContinuousUnivariateParameterState(μv)

μ.logtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test states[1].logprior == lp
@test_approx_eq states[1].logtarget lt
μ.gradlogtarget!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
@test states[1].gradlogprior == glp
@test_approx_eq states[1].gradlogtarget glt

states[1] = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test states[1].logprior == lp
@test_approx_eq states[1].logtarget lt
@test_approx_eq states[1].gradloglikelihood gll
@test states[1].gradlogprior == glp
@test_approx_eq states[1].gradlogtarget glt

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

μv = 4.21
xv = 3.1
σv = 2.
μ0v = 1.
σ0v = 10.
states = VariableState[
  ContinuousUnivariateParameterState(μv),
  UnivariateGenericVariableState(xv),
  UnivariateGenericVariableState(σv),
  UnivariateGenericVariableState(μ0v),
  UnivariateGenericVariableState(σ0v)
]

μ.prior = Normal(states[4].value, states[5].value)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states, 1)
@test_approx_eq states[1].loglikelihood ll
μ.logprior!(states, 1)
@test states[1].logprior == lp
μ.gradloglikelihood!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
μ.gradlogprior!(states, 1)
@test states[1].gradlogprior == glp

states[1] = ContinuousUnivariateParameterState(μv)

μ.logtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test states[1].logprior == lp
@test_approx_eq states[1].logtarget lt
μ.gradlogtarget!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
@test states[1].gradlogprior == glp
@test_approx_eq states[1].gradlogtarget glt

states[1] = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test states[1].logprior == lp
@test_approx_eq states[1].logtarget lt
@test_approx_eq states[1].gradloglikelihood gll
@test states[1].gradlogprior == glp
@test_approx_eq states[1].gradlogtarget glt

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood!, logprior!, gradloglikelihood! and gradlogprior! fields...")

μv = 6.69
xv = 5.43
σv = 1.
μ0v = 0.
σ0v = 1.
states = VariableState[
  ContinuousUnivariateParameterState(μv),
  UnivariateGenericVariableState(xv),
  UnivariateGenericVariableState(σv),
  UnivariateGenericVariableState(μ0v),
  UnivariateGenericVariableState(σ0v)
]

llf(states, i) =
  states[i].loglikelihood =
  -0.5*((states[2].value-states[i].value)^2/(states[3].value^2)+log(2*pi))-log(states[3].value)

lpf(states, i) =
  states[i].logprior =
  -0.5*((states[i].value-states[4].value)^2/(states[5].value^2)+log(2*pi))-log(states[5].value)

gllf(states, i) = states[i].gradloglikelihood = (states[2].value-states[i].value)/(states[3].value^2)

glpf(states, i) = states[i].gradlogprior = -(states[i].value-states[4].value)/(states[5].value^2)

μ = ContinuousUnivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf, gradloglikelihood=gllf, gradlogprior=glpf)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states, 1)
@test_approx_eq states[1].loglikelihood ll
μ.logprior!(states, 1)
@test_approx_eq states[1].logprior lp
μ.gradloglikelihood!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
μ.gradlogprior!(states, 1)
@test_approx_eq states[1].gradlogprior glp

states[1] = ContinuousUnivariateParameterState(μv)

μ.logtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test_approx_eq states[1].logprior lp
@test_approx_eq states[1].logtarget lt
μ.gradlogtarget!(states, 1)
@test_approx_eq states[1].gradloglikelihood gll
@test_approx_eq states[1].gradlogprior glp
@test_approx_eq states[1].gradlogtarget glt

states[1] = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(states, 1)
@test_approx_eq states[1].loglikelihood ll
@test_approx_eq states[1].logprior lp
@test_approx_eq states[1].logtarget lt
@test_approx_eq states[1].gradloglikelihood gll
@test_approx_eq states[1].gradlogprior glp
@test_approx_eq states[1].gradlogtarget glt

for field in [:pdf, :prior, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! and gradlogtarget! fields...")

pv = -4.29
μv = 2.2
states = VariableState[ContinuousUnivariateParameterState(pv), UnivariateGenericVariableState(μv)]

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(states, i) -> states[i].logtarget = -(states[i].value-states[2].value)^2,
  gradlogtarget=(states, i) -> states[i].gradlogtarget = -2*(states[i].value-states[2].value)
)

distribution = Normal(μv)
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test_approx_eq 0.5*(states[1].logtarget-log(2*pi)) lt
p.gradlogtarget!(states, 1)
@test_approx_eq 0.5*states[1].gradlogtarget glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test_approx_eq 0.5*(states[1].logtarget-log(2*pi)) lt
@test_approx_eq 0.5*states[1].gradlogtarget glt

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
