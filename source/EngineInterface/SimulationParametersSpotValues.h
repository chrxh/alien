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

    ColorVector<float> cellFunctionConstructorMutationNeuronDataProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationPropertiesProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationCellFunctionProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationGeometryProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationCustomGeometryProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationInsertionProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationDeletionProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationTranslationProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationDuplicationProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationCellColorProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationSubgenomeColorProbability = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<float> cellFunctionConstructorMutationGenomeColorProbability = {0, 0, 0, 0, 0, 0, 0};

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
            if (cellFunctionConstructorMutationCellColorProbability[i] != other.cellFunctionConstructorMutationCellColorProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationGenomeColorProbability[i] != other.cellFunctionConstructorMutationGenomeColorProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationGeometryProbability[i] != other.cellFunctionConstructorMutationGeometryProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationCustomGeometryProbability[i] != other.cellFunctionConstructorMutationCustomGeometryProbability[i]) {
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
            if (cellFunctionConstructorMutationNeuronDataProbability[i] != other.cellFunctionConstructorMutationNeuronDataProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationPropertiesProbability[i] != other.cellFunctionConstructorMutationPropertiesProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationCellFunctionProbability[i] != other.cellFunctionConstructorMutationCellFunctionProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationInsertionProbability[i] != other.cellFunctionConstructorMutationInsertionProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationDeletionProbability[i] != other.cellFunctionConstructorMutationDeletionProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationTranslationProbability[i] != other.cellFunctionConstructorMutationTranslationProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationDuplicationProbability[i] != other.cellFunctionConstructorMutationDuplicationProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationSubgenomeColorProbability[i] != other.cellFunctionConstructorMutationSubgenomeColorProbability[i]) {
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
