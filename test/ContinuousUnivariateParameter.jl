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
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = 6.11
nstates[:μ] = UnivariateGenericVariableState(μv)

p = ContinuousUnivariateParameter(1, :p, pdf=Normal(nstates[:μ].value))

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -11.87
pstate.value = pv
μv = -20.2
nstates[:μ].value = μv

p.pdf = Normal(nstates[:μ].value)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via prior field...")

pv = 1.25
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
σv = 10.
nstates[:σ] = UnivariateGenericVariableState(σv)

p = ContinuousUnivariateParameter(1, :p, prior=Normal(0., nstates[:σ].value))

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

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
pstate.value = pv
σv = 1.
nstates[:σ].value = σv

p.prior = Normal(0., nstates[:σ].value)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

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
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = 5.4
nstates[:μ] = UnivariateGenericVariableState(μv)

p = ContinuousUnivariateParameter(1, :p, setpdf=(pstates, nstates) -> Normal(nstates[:μ].value))
p.setpdf(pstate, nstates)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -1.91
pstate.value = pv
μv = 0.12
nstates[:μ].value = μv

p.setpdf(pstate, nstates)

distribution = Normal(μv)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setprior field...")

pv = 3.55
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
σv = 2.
nstates[:σ] = UnivariateGenericVariableState(σv)

p = ContinuousUnivariateParameter(1, :p, setprior=(pstates, nstates) -> Normal(0., nstates[:σ].value))
p.setprior(pstate, nstates)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = -2.67
pstate.value = pv
σv = 5.
nstates[:σ].value = σv

p.setprior(pstate, nstates)

distribution = Normal(0., σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood! and logprior! fields...")

μv = -2.637
pstate = ContinuousUnivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = -1.88
nstates[:x] = UnivariateGenericVariableState(xv)
σv = 1.
nstates[:σ] = UnivariateGenericVariableState(σv)
μ0v = 0.
nstates[:μ0] = UnivariateGenericVariableState(μ0v)
σ0v = 1.
nstates[:σ0] = UnivariateGenericVariableState(σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*((nstates[:x].value-pstate.value)^2/(nstates[:σ].value^2)+log(2*pi))-log(nstates[:σ].value)

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*((pstate.value-nstates[:μ0].value)^2/(nstates[:σ0].value^2)+log(2*pi))-log(nstates[:σ0].value)

μ = ContinuousUnivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
μ.loglikelihood!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
μ.logprior!(pstate, nstates)
@test_approx_eq pstate.logprior lp

pstate = ContinuousUnivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test_approx_eq pstate.logprior lp
@test_approx_eq pstate.logtarget lt

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
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = 9.4
nstates[:μ] = UnivariateGenericVariableState(μv)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)^2
)

p.logtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-log(2*pi)) logpdf(Normal(μv), pv)

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
pstate = ContinuousUnivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = 4.11
nstates[:x] = UnivariateGenericVariableState(xv)
σv = 1.
nstates[:σ] = UnivariateGenericVariableState(σv)
μ0v = 0.
nstates[:μ0] = UnivariateGenericVariableState(μ0v)
σ0v = 1.
nstates[:σ0] = UnivariateGenericVariableState(σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood = 
  -0.5*((nstates[:x].value-pstate.value)^2/(nstates[:σ].value^2)+log(2*pi))-log(nstates[:σ].value)

gllf(pstate, nstates) = pstate.gradloglikelihood = (nstates[:x].value-pstate.value)/(nstates[:σ].value^2)

μ = ContinuousUnivariateParameter(
  1,
  :μ,
  loglikelihood=llf,
  gradloglikelihood=gllf,
  prior=Normal(nstates[:μ0].value, nstates[:σ0].value)
)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
μ.logprior!(pstate, nstates)
@test pstate.logprior == lp
μ.gradloglikelihood!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
μ.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == glp

pstate = ContinuousUnivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

μv = 4.21
pstate.value = μv
xv = 3.1
nstates[:x].value = xv
σv = 2.
nstates[:σ].value = σv
μ0v = 1.
nstates[:μ0].value = μ0v
σ0v = 10.
nstates[:σ0].value = σ0v

μ.prior = Normal(nstates[:μ0].value, nstates[:σ0].value)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
μ.logprior!(pstate, nstates)
@test pstate.logprior == lp
μ.gradloglikelihood!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
μ.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == glp

pstate = ContinuousUnivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows Normal(μ0, σ0)
println("      Initialization via loglikelihood!, logprior!, gradloglikelihood! and gradlogprior! fields...")

μv = 6.69
pstate = ContinuousUnivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = 5.43
nstates[:x] = UnivariateGenericVariableState(xv)
σv = 1.
nstates[:σ] = UnivariateGenericVariableState(σv)
μ0v = 0.
nstates[:μ0] = UnivariateGenericVariableState(μ0v)
σ0v = 1.
nstates[:σ0] = UnivariateGenericVariableState(σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*((nstates[:x].value-pstate.value)^2/(nstates[:σ].value^2)+log(2*pi))-log(nstates[:σ].value)

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*((pstate.value-nstates[:μ0].value)^2/(nstates[:σ0].value^2)+log(2*pi))-log(nstates[:σ0].value)

gllf(pstate, nstates) = pstate.gradloglikelihood = (nstates[:x].value-pstate.value)/(nstates[:σ].value^2)

glpf(pstate, nstates) = pstate.gradlogprior = -(pstate.value-nstates[:μ0].value)/(nstates[:σ0].value^2)

μ = ContinuousUnivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf, gradloglikelihood=gllf, gradlogprior=glpf)

ld = Normal(μv, σv)
pd = Normal(μ0v, σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
μ.logprior!(pstate, nstates)
@test_approx_eq pstate.logprior lp
μ.gradloglikelihood!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
μ.gradlogprior!(pstate, nstates)
@test_approx_eq pstate.gradlogprior glp

pstate = ContinuousUnivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test_approx_eq pstate.logprior lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test_approx_eq pstate.gradlogprior glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousUnivariateParameterState(μv)

μ.uptogradlogtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test_approx_eq pstate.logprior lp
@test_approx_eq pstate.logtarget lt
@test_approx_eq pstate.gradloglikelihood gll
@test_approx_eq pstate.gradlogprior glp
@test_approx_eq pstate.gradlogtarget glt

for field in [:pdf, :prior, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! and gradlogtarget! fields...")

pv = -4.29
pstate = ContinuousUnivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = 2.2
nstates[:μ] = UnivariateGenericVariableState(μv)

p = ContinuousUnivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)^2,
  gradlogtarget=(pstate, nstates) -> pstate.gradlogtarget = -2*(pstate.value-nstates[:μ].value)
)

distribution = Normal(μv)
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-log(2*pi)) lt
p.gradlogtarget!(pstate, nstates)
@test_approx_eq 0.5*pstate.gradlogtarget glt

pstate = ContinuousUnivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-log(2*pi)) lt
@test_approx_eq 0.5*pstate.gradlogtarget glt

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
