using Base.Test
using Distributions
using Lora

println("    Testing ContinuousUnivariateParameter constructors...")

p = ContinuousUnivariateParameter(1, :p)
for field in (
  :pdf,
  :setpdf,
  :loglikelihood,
  :logprior,
  :logtarget,
  :gradloglikelihood,
  :gradlogprior,
  :gradlogtarget,
  :tensorloglikelihood,
  :tensorlogprior,
  :tensorlogtarget,
  :dtensorloglikelihood,
  :dtensorlogprior,
  :dtensorlogtarget,
  :uptogradlogtarget,
  :uptotensorlogtarget,
  :uptodtensorlogtarget,
  :rand
)
  @test getfield(p, field) == nothing
end

p = ContinuousUnivariateParameter(1, :p, pdf=Normal())

println("    Testing ContinuousMultivariateParameter constructors...")

ContinuousMultivariateParameter(1, :p)
ContinuousMultivariateParameter(1, :p, pdf=MvNormal(ones(2)))
