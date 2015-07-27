tests =
  [
    "states",
    "variables",
    "ContinuousUnivariateParameter",
    "ContinuousMultivariateParameter",
    "GenericModel"
  ]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
