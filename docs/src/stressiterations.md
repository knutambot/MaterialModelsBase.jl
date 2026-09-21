# Stress states
In many cases, the full 3d stress and strain states are not desired. 
In some cases, a specialized reduced material model can be written for
these cases, but for advanced models this can be tedious. A stress state
iteration procedure is therefore included in this package, since this does
not interfere with the internal implementation of the material model. 

It also allows custom implementations of specific materials for e.g. 
plane stress if desired.

The specific stress state is invoked by defining a stress state of type
`AbstractStressState`, of which the following specific stress states are implemented:

```@docs
FullStressState
PlaneStrain
PlaneStress
UniaxialStrain
UniaxialStress
UniaxialNormalStress
GeneralStressState
```

To adjust the iteration parameters (number of iterations and tolerance) for stress states
requiring iterations, such stress states contain an `IterationSettings` object,
```@docs
MaterialModelsBase.IterationSettings
```

## Reduced stress state wrapper
When used in finite element codes, it is often convenient to collect both the stress 
state and the material type into a single type that is passed to the element routine.
The wrapper `ReducedStressState` is provided for that purpose.
```@docs
ReducedStressState
```

## Postprocessing
When postprocessing an already converged simulation, it is often useful to calculate
the stress at a strain that is close to, but not exactly, the one used to produce a
given state (e.g. an interpolated value), without advancing any history variables.
`calculate_current_stress` provides this, and supports `ReducedStressState` as well.
```@docs
calculate_current_stress
```