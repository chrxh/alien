#pragma once

#include "EngineConstants.h"
#include "Colors.h"

/**
 * NOTE: header is also included in kernel code
 */

struct SimulationParametersSpotValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    ColorVector<float> radiationAbsorption = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ColorVector<float> radiationAbsorptionLowVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationAbsorptionLowGenomeComplexityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ColorVector<float> radiationCellAgeStrength = {0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f, 0.00002f};
    ColorVector<float> cellMaxForce = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f};
    ColorVector<float> cellMinEnergy = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f};
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

    bool operator==(SimulationParametersSpotValues const& other) const
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                if (cellFunctionAttackerFoodChainColorMatrix[i][j] != other.cellFunctionAttackerFoodChainColorMatrix[i][j]) {
                    return false;
                }
                if (cellFunctionAttackerGenomeComplexityBonus[i][j] != other.cellFunctionAttackerGenomeComplexityBonus[i][j]) {
                    return false;
                }
                if (cellFunctionAttackerNewComplexMutantPenalty[i][j] != other.cellFunctionAttackerNewComplexMutantPenalty[i][j]) {
                    return false;
                }
            }
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (cellMaxForce[i] != other.cellMaxForce[i]) {
                return false;
            }
            if (cellColorTransitionDuration[i] != other.cellColorTransitionDuration[i]) {
                return false;
            }
            if (cellColorTransitionTargetColor[i] != other.cellColorTransitionTargetColor[i]) {
                return false;
            }
            if (radiationAbsorptionLowGenomeComplexityPenalty[i] != other.radiationAbsorptionLowGenomeComplexityPenalty[i]) {
                return false;
            }
            if (cellCopyMutationCellColor[i] != other.cellCopyMutationCellColor[i]) {
                return false;
            }
            if (cellCopyMutationGenomeColor[i] != other.cellCopyMutationGenomeColor[i]) {
                return false;
            }
            if (cellCopyMutationGeometry[i] != other.cellCopyMutationGeometry[i]) {
                return false;
            }
            if (cellCopyMutationCustomGeometry[i] != other.cellCopyMutationCustomGeometry[i]) {
                return false;
            }
            if (radiationAbsorption[i] != other.radiationAbsorption[i]) {
                return false;
            }
            if (cellFunctionAttackerEnergyCost[i] != other.cellFunctionAttackerEnergyCost[i]) {
                return false;
            }
            if (cellFunctionAttackerGeometryDeviationExponent[i] != other.cellFunctionAttackerGeometryDeviationExponent[i]) {
                return false;
            }
            if (cellFunctionAttackerConnectionsMismatchPenalty[i] != other.cellFunctionAttackerConnectionsMismatchPenalty[i]) {
                return false;
            }
            if (cellMinEnergy[i] != other.cellMinEnergy[i]) {
                return false;
            }
            if (radiationCellAgeStrength[i] != other.radiationCellAgeStrength[i]) {
                return false;
            }
            if (cellCopyMutationNeuronData[i] != other.cellCopyMutationNeuronData[i]) {
                return false;
            }
            if (cellCopyMutationCellProperties[i] != other.cellCopyMutationCellProperties[i]) {
                return false;
            }
            if (cellCopyMutationCellFunction[i] != other.cellCopyMutationCellFunction[i]) {
                return false;
            }
            if (cellCopyMutationInsertion[i] != other.cellCopyMutationInsertion[i]) {
                return false;
            }
            if (cellCopyMutationDeletion[i] != other.cellCopyMutationDeletion[i]) {
                return false;
            }
            if (cellCopyMutationTranslation[i] != other.cellCopyMutationTranslation[i]) {
                return false;
            }
            if (cellCopyMutationDuplication[i] != other.cellCopyMutationDuplication[i]) {
                return false;
            }
            if (cellCopyMutationSubgenomeColor[i] != other.cellCopyMutationSubgenomeColor[i]) {
                return false;
            }
            if (radiationAbsorptionLowVelocityPenalty[i] != other.radiationAbsorptionLowVelocityPenalty[i]) {
                return false;
            }
            if (cellInactiveMaxAge[i] != other.cellInactiveMaxAge[i]) {
                return false;
            }
        }
        return friction == other.friction && rigidity == other.rigidity
            && cellFusionVelocity == other.cellFusionVelocity
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
        ;
    }
};
