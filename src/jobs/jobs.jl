### MCJob

abstract MCJob

Base.run{J<:MCJob}(job::Vector{J}) = map(run, job)
