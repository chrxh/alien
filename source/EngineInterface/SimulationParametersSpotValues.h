#pragma once

#include "Constants.h"

//NOTE: header is also included in CUDA code
using FloatByColor = float[MAX_COLORS];
using IntByColor = int[MAX_COLORS];
struct SimulationParametersSpotValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    FloatByColor radiationCellAgeStrength = {0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f};
    float cellMaxForce = 0.8f;
    FloatByColor cellMinEnergy = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f};
    float cellFusionVelocity = 0.4f;
    float cellMaxBindingEnergy = 500000.0f;
    IntByColor cellColorTransitionDuration = {0, 0, 0, 0, 0, 0, 0};
    IntByColor cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};
    float cellFunctionAttackerEnergyCost = 0.0f;
    float cellFunctionAttackerFoodChainColorMatrix[MAX_COLORS][MAX_COLORS] = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}};

    float cellFunctionAttackerGeometryDeviationExponent = 0.0f;
    float cellFunctionAttackerConnectionsMismatchPenalty = 0.33f;

    FloatByColor cellFunctionConstructorMutationNeuronDataProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationDataProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationCellFunctionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationInsertionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationDeletionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationTranslationProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor cellFunctionConstructorMutationDuplicationProbability = {0, 0, 0, 0, 0, 0, 0};

    bool operator==(SimulationParametersSpotValues const& other) const
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                if (cellFunctionAttackerFoodChainColorMatrix[i][j] != other.cellFunctionAttackerFoodChainColorMatrix[i][j]) {
                    return false;
                }
            }
            if (cellColorTransitionDuration[i] != other.cellColorTransitionDuration[i]) {
                return false;
            }
            if (cellColorTransitionTargetColor[i] != other.cellColorTransitionTargetColor[i]) {
                return false;
            }
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (cellMinEnergy[i] != other.cellMinEnergy[i]) {
                return false;
            }
            if (radiationCellAgeStrength[i] != other.radiationCellAgeStrength[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationNeuronDataProbability[i] != other.cellFunctionConstructorMutationNeuronDataProbability[i]) {
                return false;
            }
            if (cellFunctionConstructorMutationDataProbability[i] != other.cellFunctionConstructorMutationDataProbability[i]) {
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
        }
        return friction == other.friction && rigidity == other.rigidity && cellMaxForce == other.cellMaxForce
            && cellFusionVelocity == other.cellFusionVelocity
            && cellFunctionAttackerEnergyCost == other.cellFunctionAttackerEnergyCost
            && cellFunctionAttackerGeometryDeviationExponent == other.cellFunctionAttackerGeometryDeviationExponent
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
            && cellFunctionAttackerConnectionsMismatchPenalty == other.cellFunctionAttackerConnectionsMismatchPenalty
        ;
    }
};
