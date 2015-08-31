### Abstract variable IOStreams

abstract VariableIOStream{F<:VariateForm, N<:Number}

abstract GenericVariableIOStream{F<:VariateForm, N<:Number} <: VariableIOStream{F, N}

abstract ParameterIOStream{F<:VariateForm, N<:Number} <: VariableIOStream{F, N}

Base.eltype{F<:VariateForm, N<:Number}(::Type{VariableIOStream{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{GenericVariableIOStream{F, N}}) = N
Base.eltype{F<:VariateForm, N<:Number}(::Type{ParameterIOStream{F, N}}) = N
