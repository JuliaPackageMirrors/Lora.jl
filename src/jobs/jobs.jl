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

# initialize_output() needs to be defined for custom variable state or NState input arguments
# Thus multiple dispatch allows to extend the code base to accommodate new variable states or NStates

function initialize_output(
  state::ContinuousUnivariateParameterState,
  n::Int,
  outopts::Dict{Symbol, Any}
)
  output::Union{VariableNState, VariableIOStream, Void}

  output = if outopts[:destination] == :nstate
    ContinuousUnivariateParameterNState(eltype(state), n, outopts[:monitor], outopts[:diagnostics])
  elseif outopts[:destination] == :iostream
    ContinuousParameterIOStream(
      "w",
      (),
      n,
      outopts[:monitor];
      diagnostickeys=outopts[:diagnostics],
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix]
    )
  else
    error("Error message")
  end

  output
end

Base.run{J<:MCJob}(job::Vector{J}) = map(run, job)
