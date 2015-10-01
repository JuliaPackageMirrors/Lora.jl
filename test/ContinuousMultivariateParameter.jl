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

println("    Testing ContinuousMultivariateParameter constructors:")

println("      Initialization via index and key fields...")

p = ContinuousMultivariateParameter(1, :p)

for field in values(fields)
  @test getfield(p, field) == nothing
end

println("      Initialization via pdf field...")

pv = [5.18, -7.76]
μv = [6.11, -8.5]
states = VariableState[ContinuousMultivariateParameterState(pv), MultivariateGenericVariableState(μv)]

p = ContinuousMultivariateParameter(1, :p, pdf=MvNormal(states[2].value, 1.))

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states, 1)
@test states[1].logtarget == lt
p.gradlogtarget!(states, 1)
@test states[1].gradlogtarget == glt

states[1] = ContinuousMultivariateParameterState(pv)

p.uptogradlogtarget!(states, 1)
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-11.87, -13.44]
pstate.value = pv
μv = [-20.2, -18.91]
nstates[:μ].value = μv

p.pdf = MvNormal(nstates[:μ].value, 1.)

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousMultivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via prior field...")

pv = [1.25, 1.8]
pvlen = length(pv)
pstate = ContinuousMultivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
σv = [10., 2.]
nstates[:σ] = MultivariateGenericVariableState(σv)

p = ContinuousMultivariateParameter(1, :p, prior=MvNormal(zeros(pvlen), nstates[:σ].value))

distribution = MvNormal(zeros(pvlen), σv)
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

pv = [-0.21, 0.98]
pvlen = length(pv)
pstate.value = pv
σv = ones(pvlen)
nstates[:σ].value = σv

p.prior = MvNormal(zeros(pvlen), nstates[:σ].value)

distribution = MvNormal(zeros(pvlen), σv)
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

pv = [3.79, 4.64]
pstate = ContinuousMultivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = [5.4, 5.3]
nstates[:μ] = MultivariateGenericVariableState(μv)

p = ContinuousMultivariateParameter(1, :p, setpdf=(pstates, nstates) -> MvNormal(nstates[:μ].value, 1.))
p.setpdf(pstate, nstates)

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousMultivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-1.91, -0.9]
pstate.value = pv
μv = [0.12, 0.99]
nstates[:μ].value = μv

p.setpdf(pstate, nstates)

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test pstate.logtarget == lt
p.gradlogtarget!(pstate, nstates)
@test pstate.gradlogtarget == glt

pstate = ContinuousMultivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test (pstate.logtarget, pstate.gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setprior field...")

pv = [3.55, 9.5]
pvlen = length(pv)
pstate = ContinuousMultivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
σv = [2., 10.]
nstates[:σ] = MultivariateGenericVariableState(σv)

p = ContinuousMultivariateParameter(1, :p, setprior=(pstates, nstates) -> MvNormal(zeros(pvlen), nstates[:σ].value))
p.setprior(pstate, nstates)

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-2.67, 7.71]
pstate.value = pv
σv = [5., 4.]
nstates[:σ].value = σv

p.setprior(pstate, nstates)

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(pstate, nstates)
@test pstate.logprior == logpdf(distribution, pv)
p.gradlogprior!(pstate, nstates)
@test pstate.gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood! and logprior! fields...")

μv = [-2.637, -1.132]
μvlen = length(μv)
pstate = ContinuousMultivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = [-1.88, 2.23]
nstates[:x] = MultivariateGenericVariableState(xv)
Σv = eye(μvlen)
nstates[:Σ] = MatrixvariateGenericVariableState(Σv)
μ0v = zeros(μvlen)
nstates[:μ0] = MultivariateGenericVariableState(μ0v)
Σ0v = eye(μvlen)
nstates[:Σ0] = MatrixvariateGenericVariableState(Σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*(
    (nstates[:x].value-pstate.value)'*inv(nstates[:Σ].value)*(nstates[:x].value-pstate.value)+
    μvlen*log(2*pi)+
    logdet(nstates[:Σ].value)
  )[1]

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*(
    (pstate.value-nstates[:μ0].value)'*inv(nstates[:Σ0].value)*(pstate.value-nstates[:μ0].value)+
    μvlen*log(2*pi)+
    logdet(nstates[:Σ0].value)
  )[1]

μ = ContinuousMultivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
μ.loglikelihood!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
μ.logprior!(pstate, nstates)
@test_approx_eq pstate.logprior lp

pstate = ContinuousMultivariateParameterState(μv)

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

pv = [-1.28, 1.73]
pvlen = length(pv)
pstate = ContinuousMultivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = [9.4, 3.32]
nstates[:μ] = MultivariateGenericVariableState(μv)

p = ContinuousMultivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)⋅(pstate.value-nstates[:μ].value)
)

p.logtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-pvlen*log(2*pi))[1] logpdf(MvNormal(μv, 1.), pv)

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

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood!, gradloglikelihood! and prior fields...")

μv = [5.59, -7.25]
μvlen = length(μv)
pstate = ContinuousMultivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = [4.11, 8.17]
nstates[:x] = MultivariateGenericVariableState(xv)
Σv = eye(μvlen)
nstates[:Σ] = MatrixvariateGenericVariableState(Σv)
μ0v = zeros(μvlen)
nstates[:μ0] = MultivariateGenericVariableState(μ0v)
Σ0v = eye(μvlen)
nstates[:Σ0] = MatrixvariateGenericVariableState(Σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*(
    (nstates[:x].value-pstate.value)'*inv(nstates[:Σ].value)*(nstates[:x].value-pstate.value)+
    μvlen*log(2*pi)+
    logdet(nstates[:Σ].value)
  )[1]

gllf(pstate, nstates) = pstate.gradloglikelihood = (nstates[:Σ].value\(nstates[:x].value-pstate.value))

μ = ContinuousMultivariateParameter(
  1,
  :μ,
  loglikelihood=llf,
  gradloglikelihood=gllf,
  prior=MvNormal(nstates[:μ0].value, nstates[:Σ0].value)
)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
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

pstate = ContinuousMultivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousMultivariateParameterState(μv)

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

μv = [4.21, 7.91]
pstate.value = μv
xv = [-3.1, -2.52]
nstates[:x].value = xv
Σv = diagm([2., 1.])
nstates[:Σ].value = Σv
μ0v = [1., 2.5]
nstates[:μ0].value = μ0v
Σ0v = diagm([3., 5.])
nstates[:Σ0].value = Σ0v

μ.prior = MvNormal(nstates[:μ0].value, nstates[:Σ0].value)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
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

pstate = ContinuousMultivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test pstate.logprior == lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test pstate.gradlogprior == glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousMultivariateParameterState(μv)

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

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood!, logprior!, gradloglikelihood! and gradlogprior! fields...")

μv = [6.69, -3.125]
μvlen = length(μv)
pstate = ContinuousMultivariateParameterState(μv)
nstates = Dict{Symbol, VariableState}()
xv = [5.43, 9.783]
nstates[:x] = MultivariateGenericVariableState(xv)
Σv = eye(μvlen)
nstates[:Σ] = MatrixvariateGenericVariableState(Σv)
μ0v = zeros(μvlen)
nstates[:μ0] = MultivariateGenericVariableState(μ0v)
Σ0v = eye(μvlen)
nstates[:Σ0] = MatrixvariateGenericVariableState(Σ0v)

llf(pstate, nstates) =
  pstate.loglikelihood =
  -0.5*(
    (nstates[:x].value-pstate.value)'*inv(nstates[:Σ].value)*(nstates[:x].value-pstate.value)+
    μvlen*log(2*pi)+
    logdet(nstates[:Σ].value)
  )[1]

lpf(pstate, nstates) =
  pstate.logprior =
  -0.5*(
    (pstate.value-nstates[:μ0].value)'*inv(nstates[:Σ0].value)*(pstate.value-nstates[:μ0].value)+
    μvlen*log(2*pi)+
    logdet(nstates[:Σ0].value)
  )[1]

gllf(pstate, nstates) = pstate.gradloglikelihood = (nstates[:Σ].value\(nstates[:x].value-pstate.value))

glpf(pstate, nstates) = pstate.gradlogprior = -(nstates[:Σ0].value\(pstate.value-nstates[:μ0].value))

μ = ContinuousMultivariateParameter(1, :μ, loglikelihood=llf, logprior=lpf, gradloglikelihood=gllf, gradlogprior=glpf)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
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

pstate = ContinuousMultivariateParameterState(μv)

μ.logtarget!(pstate, nstates)
@test_approx_eq pstate.loglikelihood ll
@test_approx_eq pstate.logprior lp
@test_approx_eq pstate.logtarget lt
μ.gradlogtarget!(pstate, nstates)
@test_approx_eq pstate.gradloglikelihood gll
@test_approx_eq pstate.gradlogprior glp
@test_approx_eq pstate.gradlogtarget glt

pstate = ContinuousMultivariateParameterState(μv)

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

pv = [-4.29, 2.91]
pstate = ContinuousMultivariateParameterState(pv)
nstates = Dict{Symbol, VariableState}()
μv = [2.2, 2.02]
nstates[:μ] = MultivariateGenericVariableState(μv)

p = ContinuousMultivariateParameter(
  1,
  :p,
  logtarget=(pstate, nstates) -> pstate.logtarget = -(pstate.value-nstates[:μ].value)⋅(pstate.value-nstates[:μ].value),
  gradlogtarget=(pstate, nstates) -> pstate.gradlogtarget = -2*(pstate.value-nstates[:μ].value)
)

distribution = MvNormal(μv, 1.)
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-pvlen*log(2*pi))[1] lt
p.gradlogtarget!(pstate, nstates)
@test_approx_eq 0.5*pstate.gradlogtarget glt

pstate = ContinuousMultivariateParameterState(pv)

p.uptogradlogtarget!(pstate, nstates)
@test_approx_eq 0.5*(pstate.logtarget-pvlen*log(2*pi))[1] lt
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
