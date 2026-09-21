"""
    calculate_current_stress(m::AbstractMaterial, strain, state::AbstractMaterialState)
    calculate_current_stress(stress_state::AbstractStressState, m::AbstractMaterial, strain, state::AbstractMaterialState)
    calculate_current_stress(rss::ReducedStressState, strain, state::AbstractMaterialState)

## Using this interface
Calculate the stress that is energy-conjugated to `strain`, consistent with the *given*
`state`, without invoking any local iteration that would advance history/internal
variables. `state` is normally the already-converged state obtained from a previous
call to `material_response`, e.g. during postprocessing.

## Implementing this interface
A material-model developer only needs to implement the full-dimensional method,
`calculate_current_stress(m::MyMaterial, strain, state::MyMaterialState)`. If `MyMaterial`
has no state (i.e. `initial_material_state(m) isa NoMaterialState`), this is not
required either, since [`material_response`](@ref) is then already frozen-state by
definition and a generic fallback is provided. Support for a reduced-dimensional stress 
state (e.g. via [`ReducedStressState`](@ref)) then follows automatically from a generic 
fallback, using the tangent obtained by automatic differentiation via `Tensors.gradient`. 
A specific reduced-dimensional method, 
`calculate_current_stress(stress_state::AbstractStressState, m::MyMaterial, strain, state::MyMaterialState)`,
can be added when a cheaper, non-autodiff alternative exists.
"""
function calculate_current_stress end

# Fully generic: a material with no state has, by definition, nothing to freeze -
# `material_response` already gives the frozen-state stress.
function calculate_current_stress(m::AbstractMaterial, strain, state::NoMaterialState)
    σ, _, _ = material_response(m, strain, state)
    return σ
end

# Wraps a material `m` and a frozen state `s` as an `AbstractMaterial`, whose
# `material_response` evaluates `calculate_current_stress(m, strain, s)` (at fixed
# history/internal variables) so that it can ride the existing stress-state Newton
# iteration (e.g. for `PlaneStress`). The tangent needed for that iteration is
# obtained via automatic differentiation. This powers the generic reduced-dimensional
# fallback of `calculate_current_stress` below.
struct FrozenStressMaterial{MT <: AbstractMaterial, ST <: AbstractMaterialState} <: AbstractMaterial
    m::MT
    s::ST
end
function material_response(fm::FrozenStressMaterial, strain::SecondOrderTensor{3}, old::AbstractMaterialState, args...)
    dσdϵ, σ = Tensors.gradient(e -> calculate_current_stress(fm.m, e, fm.s), strain, :all)
    return σ, dσdϵ, old
end

# Generic reduced-dimensional fallback: as long as `calculate_current_stress(m, strain,
# state)` (full-dimensional) is implemented for `m`, this makes `ReducedStressState`
# support "just work", by autodiff-ing through it. The `NoMaterialState` fast path
# below takes precedence when a cheaper, non-autodiff alternative exists.
function calculate_current_stress(stress_state::AbstractStressState, m::AbstractMaterial, strain, state::AbstractMaterialState)
    frozen = FrozenStressMaterial(m, state)
    σ, _, _, _ = material_response(stress_state, frozen, strain, NoMaterialState{eltype(strain)}())
    return σ
end

# Reduced-dimensional fast path for stateless materials: avoids the autodiff in the
# generic fallback above by delegating directly to `material_response`'s own
# (potentially analytic) stress-state handling.
function calculate_current_stress(stress_state::AbstractStressState, m::AbstractMaterial, strain, state::NoMaterialState)
    return first(material_response(stress_state, m, strain, state))
end

function calculate_current_stress(rss::ReducedStressState, strain, state::AbstractMaterialState)
    return calculate_current_stress(rss.stress_state, rss.material, strain, state)
end

# Disambiguates the two 3-argument methods above for a `ReducedStressState` wrapping a
# stateless material.
function calculate_current_stress(rss::ReducedStressState, strain, state::NoMaterialState)
    return calculate_current_stress(rss.stress_state, rss.material, strain, state)
end
