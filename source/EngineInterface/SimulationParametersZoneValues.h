#pragma once

#include "EngineConstants.h"
#include "Colors.h"

/**
 * NOTE: header is also included in kernel code
 */

struct SimulationParametersZoneValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    ColorVector<float> radiationAbsorption = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> radiationAbsorptionLowVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationAbsorptionLowGenomeComplexityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationCellAgeStrength = {0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f};
    bool radiationDisableSources = false;
    ColorVector<float> cellMaxForce = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f};
    ColorVector<float> cellMinEnergy = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f};
    ColorVector<float> cellDeathProbability = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};

    float cellFusionVelocity = 0.6f;
    float cellMaxBindingEnergy = Infinity<float>::value;
    ColorVector<float> cellInactiveMaxAge = {
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};
    ColorVector<int> cellColorTransitionDuration = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    ColorVector<int> cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};
    ColorVector<float> cellFunctionAttackerEnergyCost = {0, 0, 0, 0, 0, 0, 0};
    ColorMatrix<float> cellFunctionAttackerFoodChainColorMatrix = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}};

    ColorVector<float> cellFunctionAttackerGeometryDeviationExponent = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionAttackerConnectionsMismatchPenalty = {0, 0, 0, 0, 0, 0, 0};
    ColorMatrix<float> cellFunctionAttackerGenomeComplexityBonus = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    ColorMatrix<float> cellFunctionAttackerNewComplexMutantPenalty = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    ColorVector<float> cellCopyMutationNeuronData = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationCellProperties = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationCellFunction = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationGeometry = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationCustomGeometry = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationInsertion = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationDeletion = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationTranslation = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationDuplication = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationCellColor = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationSubgenomeColor = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellCopyMutationGenomeColor = {0, 0, 0, 0, 0, 0, 0};

    bool operator==(SimulationParametersZoneValues const&) const = default;
};
