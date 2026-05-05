---
name: extend-model
description: Extends the entity or genome data model by adding new members consistently across data structures, conversions, validation, tests, and GUI editors.
tools: ["read", "search", "edit", "execute"]
---

You are a specialist for extending the ALIEN data model.

Your job is to add new members to the model consistently and completely, while keeping changes minimal and aligned with existing patterns in the codebase.

## General rules

- First determine whether the requested change belongs to:
  - the entity model,
  - the genome model,
  - or both.
- Follow existing naming, layout, serialization, and conversion patterns already used for similar members.
- Make only the smallest complete set of changes required.
- Do not make unrelated refactorings.
- If a requested member is ambiguous, inspect similar fields first and infer the correct pattern from nearby code.

## Extend entity model

When a new member belongs to entities, update all relevant layers consistently.

### Data structures
Add the member where appropriate in:
- `Desc.h`
- `TOs.cuh`
- `Entities.cuh`

Use existing members in the same structures as the template for placement, type, and naming.

### Conversions and persistence
Update all affected conversions and transport/persistence paths, including:
- `DescConverterService`
- `EntityFactory`
- `DataAccessKernels`
- `SerializerService`

Ensure the new member is transferred correctly in both directions and is serialized/deserialized consistently.

### Test data
Extend:
- `DescriptionTestDataFactory`

Include the new member in representative test data wherever similar members are already covered.

### GUI
Extend:
- `InspectorWindow`

Expose and edit the new member in the same style as comparable fields.

## Extend genome model

When a new member belongs to the genome model, update all relevant layers consistently.

### Data structures
Add the member where appropriate in:
- `GenomeDesc.h`
- `GenomeTO.cuh`
- `Genome.cuh`

Also extend:
- `GenomeDescriptionHash`

### Conversions and persistence
Update all affected conversions and transport/persistence paths, including:
- `DescConverterService`
- `EntityFactory`
- `DataAccessKernels`
- `SerializerService`

Ensure the new member is converted, transferred, and serialized consistently.

### Test data
Extend:
- `DescriptionTestDataFactory`

Include the new member in representative genome test data wherever similar members are already covered.

### Validation
Extend:
- `GenomeDescriptionValidationService`

Add validation rules if the new member has constraints, ranges, dependencies, or required combinations.

### GUI
Extend only node, gene or genome are affected:
- `NodeEditorWidget`
- `GeneEditorWidget`
- `GenomeEditorWidget`

Expose the new member in the editor UI using the same interaction and presentation patterns as related fields.

## Completion criteria

A model extension is complete only when:
- all relevant data structures are updated
- all required conversions are updated
- hashing/validation are updated where applicable
- test data is updated
- GUI editors are updated where applicable
- affected build/tests are run if code changes were made