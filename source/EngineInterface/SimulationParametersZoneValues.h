#pragma once

#include "EngineConstants.h"
#include "Colors.h"

/**
 * NOTE: header is also included in kernel code
 */

struct ColorTransitionRules
{
    ColorVector<int> cellColorTransitionDuration = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    ColorVector<int> cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};

    bool operator==(ColorTransitionRules const&) const = default;
};

struct SimulationParametersZoneValues
{
    uint32_t backgroundColor = 0x1b0000;
    float friction = 0.001f;
    float rigidity = 0.0f;
    ColorVector<float> radiationAbsorption = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> radiationAbsorptionLowVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationAbsorptionLowGenomeComplexityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationType1_strength = {0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f};
    bool radiationDisableSources = false;
    ColorVector<float> cellMaxForce = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f};
    ColorVector<float> minCellEnergy = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f};
    ColorVector<float> cellDeathProbability = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};

    float cellFusionVelocity = 2.0f;
    float cellMaxBindingEnergy = Infinity<float>::value;
    ColorVector<float> maxAgeForInactiveCells = {   // Candidate for deletion
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};
    ColorTransitionRules colorTransitionRules;
    ColorVector<float> attackerEnergyCost = {0, 0, 0, 0, 0, 0, 0};
    ColorMatrix<float> attackerFoodChainColorMatrix = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}};

    ColorVector<float> attackerGeometryDeviationProtection = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> attackerConnectionsMismatchProtection = {0, 0, 0, 0, 0, 0, 0};
    ColorMatrix<float> attackerCreatureProtection = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    ColorMatrix<float> attackerNewComplexMutantProtection = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    ColorVector<float> copyMutationNeuronData = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationCellProperties = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationCellType = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationGeometry = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationCustomGeometry = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationInsertion = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationDeletion = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationTranslation = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationDuplication = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationCellColor = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationSubgenomeColor = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> copyMutationGenomeColor = {0, 0, 0, 0, 0, 0, 0};

    bool operator==(SimulationParametersZoneValues const&) const = default;
};
