### BasicMCRunner

# BasicMCRunner is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) runner
# It contains nagivational info (burnin, thinning, number of steps) and functions for managing output of simulation

immutable BasicMCRunner <: MCRunner
  burnin::Int
  thinning::Int
  nsteps::Int
  r::Range{Int}

  function BasicMCRunner(burnin::Int, thinning::Int, nsteps::Int)
    @assert burnin >= 0 "Number of burn-in iterations should be non-negative"
    @assert thinning >= 1 "Thinning should be >= 1"
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations"
    new(burnin, thinning, nsteps, (burnin+1):thinning:nsteps)
  end
end

BasicMCRunner(; burnin::Int=0, thinning::Int=1, nsteps::Int=100) = BasicMCRunner(burnin, thinning, nsteps)

BasicMCRunner(r::Range{Int}) = BasicMCRunner(first(r)-1, r.step, last(r))

BasicMCRunner(r::UnitRange{Int}) = BasicMCRunner(first(r)-1, 1, last(r))
