tests =
  [
    "parameters",
    "states",
    "GenericModel"
  ]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
