### MCJob

abstract MCJob

# Set defaults for possibly unspecified output options

function augment!(outopts::Dict{Symbol, Any})
  destination = get!(outopts, :destination, :nstate)

  if destination != :none
    if !haskey(outopts, :monitor)
      outopts[:monitor] = [:value]
    end

    if !haskey(outopts, :diagnostics)
      outopts[:diagnostics] = Symbol[]
    end

    if destination == :iostream
      if !haskey(outopts, :filepath)
        outopts[:filepath] = ""
      end

      if !haskey(outopts, :filesuffix)
        outopts[:filesuffix] = "csv"
      end
    end
  end
end

augment!(outopts::Vector{Dict{Symbol, Any}}) = map(augment!, outopts)

Base.run{J<:MCJob}(job::Vector{J}) = map(run, job)
