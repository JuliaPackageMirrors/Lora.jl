### BasicMCRunner

# BasicMCRunner is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) runner
# It contains nagivational info (burnin, thinning, number of steps)

immutable BasicMCRunner <: MCRunner
  burnin::Int
  thinning::Int
  nsteps::Int
  postrange::Range{Int}

  function BasicMCRunner(postrange::Range{Int})
    burnin = first(postrange)-1
    thinning = postrange.step
    nsteps = last(postrange)

    @assert burnin >= 0 "Number of burn-in iterations should be non-negative"
    @assert thinning >= 1 "Thinning should be >= 1"
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations"

    new(burnin, thinning, nsteps, postrange)
  end
end

BasicMCRunner(postrange::UnitRange{Int}) = BasicMCRunner(first(postrange):1:last(postrange))

BasicMCRunner(; burnin::Int=0, thinning::Int=1, nsteps::Int=100) = BasicMCRunner((burnin+1):thinning:nsteps)
