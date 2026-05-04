---
name: Extend-Model-Agent
description: Extends the data model bei new members
---

# Extend entity model

## Extend data structures
If the new member should be added to the entities, then add them to the files Desc.h, TOs.cuh and Entities.cuh. Look at other members there for orientation.

## Extend conversions
Add conversions to DescConverterService, EntityFactory, DataAccessKernels and SerializerService for new members.

## Extend TestDataFactory
Extend DescriptionTestDataFactory for new members.

## GUI
Extend InspectorWindow for new members.

# Extend genome model

## Extend data structures
If the new members should be added to the genome model, then add them to the files GenomeDesc.h, GenomeTO.cuh and Genome.cuh. Extend GenomeDescriptionHash.

## Extend conversions
Add conversions to DescConverterService, EntityFactory, DataAccessKernels and SerializerService for new members.

## Extend TestDataFactory
Extend DescriptionTestDataFactory for new members.

## Extend validation
Add validations to GenomeDescriptionValidationService.

## GUI
Extend NodeEditorWidget, GeneEditorWidget and GenomeEditorWidget for new members.
