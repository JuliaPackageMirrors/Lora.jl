### BasicMCRange

# BasicMCRange is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) range
# It contains nagivational info (burnin, thinning, number of steps)

immutable BasicMCRange{T<:Int} <: MCRange{T}
  burnin::T
  thinning::T
  nsteps::T
  postrange::Range{T}
  npoststeps::T

  function BasicMCRange(postrange::Range{Int})
    burnin = first(postrange)-1
    thinning = postrange.step
    nsteps = last(postrange)

    @assert burnin >= 0 "Number of burn-in iterations should be non-negative"
    @assert thinning >= 1 "Thinning should be >= 1"
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations"

    npoststeps = length(postrange)

    new(burnin, thinning, nsteps, postrange, npoststeps)
  end
end

BasicMCRange(postrange::UnitRange{Int}) = BasicMCRange(first(postrange):1:last(postrange))

BasicMCRange(; burnin::Int=0, thinning::Int=1, nsteps::Int=100) = BasicMCRange((burnin+1):thinning:nsteps)
