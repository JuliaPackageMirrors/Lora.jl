@doc doc"""
The `Sampleability` type is used for setting a *Variable* object to be *Constant*, *Deterministic* or *Random*.

* `Constant`: A *Constant* *Variable* has a fixed value, therefore it is a trivial case of variable. Examples of it
include its dimensions, model size and hyper-parameters.
* `Deterministic`: As opposed to a *Constant*, the value of a *Deterministic* *Variable* can vary, though it is fully
determined given other variables.
* `Random`: A *Random* *Variable* is associated with a (possibly unnormalized) distribution. Its value is randomly
determined either by sampling directly from its distribution or indirectly via Markov Chain Monte Carlo (MCMC) sampling.
"""->
abstract Sampleability

@doc doc"""
For more information on `Constant`, see documentation on *Sampleability*.
"""->
type Constant <: Sampleability end

@doc doc"""
For more information on `Deterministic`, see documentation on *Sampleability*.
"""->
type Deterministic <: Sampleability end

@doc doc"""
For more information on `Random`, see documentation on *Sampleability*.
"""->
type Random <: Sampleability end

@doc doc"""
Instances of `Variable` are the building blocks of a *Model*. For example, a *Variable* can be used for defining
hyper-parameters, data, deterministic or random variables in a *Model*.

A *Model* can be seen as a graph, whose vertices are instances of *Variable* and edges connect dependent *Variable*
nodes. In fact, it is possible to convert a *Model* to a graph as an instance of the *GenericGraph* type of the
*Graphs* package.

A *Model* is a parametric type parameterized by:

* *VaritateForm*: Abstract type with sub-types *Univariate*, *Multivariate* and *Matrixvariate*. See *Distributions*
package.
* *ValueSupport*: Abstract type with sub-types *Discrete* and *Continuous*. See *Distributions* package.
* *Sampleability*: Abstract type with sub-types *Constant*, *Deterministic* and *Random*. See documentation on
*Sampleability*.
"""->
abstract Variable{F<:VariateForm, S<:ValueSupport, A<:Sampleability}

typealias Data{F<:VariateForm, S<:ValueSupport} Variable{F, S, Constant}

typealias Hyperparameter{F<:VariateForm, S<:ValueSupport} Variable{F, S, Constant}

typealias Transformation{F<:VariateForm, S<:ValueSupport} Variable{F, S, Deterministic}

typealias Parameter{F<:VariateForm, S<:ValueSupport} Variable{F, S, Random}

typealias Dependence Edge{Variable}
