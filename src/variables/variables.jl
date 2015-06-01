@doc doc"""
The `Sampleability` type is used for setting a *Variable* object to be *Deterministic* or *Random*.

* `Deterministic`: The value of a *Deterministic* *Variable* can vary, though it is fully determined given other
variables.
* `Random`: A *Random* *Variable* is associated with a (possibly unnormalized) distribution. Its value is randomly
determined either by sampling directly from its distribution or indirectly via Markov Chain Monte Carlo (MCMC) sampling.
"""->
abstract Sampleability

@doc doc"""
For more information on `Deterministic`, see documentation on *Sampleability*.
"""->
type Deterministic <: Sampleability end

@doc doc"""
For more information on `Random`, see documentation on *Sampleability*.
"""->
type Random <: Sampleability end

abstract VariableState{F<:VariateForm, N<:Number, S<:Sampleability}

@doc doc"""
Instances of `Variable` are the building blocks of a *Model*. For example, a *Variable* can be used for defining
hyper-parameters, data, deterministic or random variables in a *Model*.

A *Model* can be seen as a graph, whose vertices are instances of *Variable* and edges connect dependent *Variable*
nodes. In fact, it is possible to convert a *Model* to a graph as an instance of the *GenericGraph* type of the
*Graphs* package.

A *Variable* is a parametric type parameterized by:

* *VariateForm*: Abstract type with sub-types *Univariate*, *Multivariate* and *Matrixvariate*. See *Distributions*
package.
* *Number*: Numeric type of *Variable*. See documentation of Julia on its builtin number system. 
* *Sampleability*: Abstract type with sub-types *Deterministic* and *Random*. See documentation on *Sampleability*.
"""->
abstract Variable{F<:VariateForm, N<:Number, S<:Sampleability}

vertex_index{V<:Variable}(v::V) = v.index
