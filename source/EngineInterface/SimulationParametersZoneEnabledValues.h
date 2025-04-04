#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersZoneEnabledValues
{
    bool rigidity = false;
    bool radiationAbsorptionLowVelocityPenalty = false;
    bool radiationAbsorptionLowGenomeComplexityPenalty = false;
    bool radiationDisableSources = false;
    bool maxAgeForInactiveCellsEnabled = false;
    bool colorTransitionRules = false;
    bool attackerEnergyCost = false;
    bool attackerFoodChainColorMatrix = false;
    bool attackerComplexCreatureProtection = false;
    bool attackerNewComplexMutantProtection = false;
    bool attackerGeometryDeviationProtection = false;
    bool attackerConnectionsMismatchProtection = false;

    bool operator==(SimulationParametersZoneEnabledValues const&) const = default;
};
