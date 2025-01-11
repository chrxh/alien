#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersZoneActivatedValues
{
    bool friction = false;
    bool rigidity = false;
    bool radiationAbsorption = false;
    bool radiationAbsorptionLowVelocityPenalty = false;
    bool radiationAbsorptionLowGenomeComplexityPenalty = false;
    bool radiationCellAgeStrength = false;
    bool radiationDisableSources = false;
    bool cellMaxForce = false;
    bool cellMinEnergy = false;
    bool cellDeathProbability = false;
    bool cellFusionVelocity = false;
    bool cellMaxBindingEnergy = false;
    bool cellInactiveMaxAge = false;
    bool cellColorTransition = false;
    bool cellTypeAttackerEnergyCost = false;
    bool cellTypeAttackerFoodChainColorMatrix = false;
    bool cellTypeAttackerGenomeComplexityBonus = false;
    bool cellTypeAttackerNewComplexMutantPenalty = false;
    bool cellTypeAttackerGeometryDeviationExponent = false;
    bool cellTypeAttackerConnectionsMismatchPenalty = false;

    bool cellCopyMutationNeuronData = false;
    bool cellCopyMutationCellProperties = false;
    bool cellCopyMutationCellType = false;
    bool cellCopyMutationGeometry = false;
    bool cellCopyMutationCustomGeometry = false; 
    bool cellCopyMutationInsertion = false;
    bool cellCopyMutationDeletion = false;
    bool cellCopyMutationTranslation = false;
    bool cellCopyMutationDuplication = false;
    bool cellCopyMutationCellColor = false;
    bool cellCopyMutationSubgenomeColor = false;
    bool cellCopyMutationGenomeColor = false;

    bool operator==(SimulationParametersZoneActivatedValues const&) const = default;
};
