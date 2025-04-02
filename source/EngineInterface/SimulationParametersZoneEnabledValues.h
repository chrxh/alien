#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersZoneEnabledValues
{
    bool rigidity = false;
    bool radiationAbsorption = false;
    bool radiationAbsorptionLowVelocityPenalty = false;
    bool radiationAbsorptionLowGenomeComplexityPenalty = false;
    bool radiationType1_strength = false;
    bool radiationDisableSources = false;
    bool minCellEnergy = false;
    bool cellDeathProbability = false;
    bool maxAgeForInactiveCellsEnabled = false;
    bool colorTransitionRules = false;
    bool attackerEnergyCost = false;
    bool attackerFoodChainColorMatrix = false;
    bool attackerComplexCreatureProtection = false;
    bool attackerNewComplexMutantProtection = false;
    bool attackerGeometryDeviationProtection = false;
    bool attackerConnectionsMismatchProtection = false;

    bool copyMutationNeuronData = false;
    bool copyMutationCellProperties = false;
    bool copyMutationCellType = false;
    bool copyMutationGeometry = false;
    bool copyMutationCustomGeometry = false; 
    bool copyMutationInsertion = false;
    bool copyMutationDeletion = false;
    bool copyMutationTranslation = false;
    bool copyMutationDuplication = false;
    bool copyMutationCellColor = false;
    bool copyMutationSubgenomeColor = false;
    bool copyMutationGenomeColor = false;

    bool operator==(SimulationParametersZoneEnabledValues const&) const = default;
};
