#pragma once

#include <limits>

#include "Constants.h"

//NOTE: header is also included in CUDA code
using FloatColorVector = float[MAX_COLORS];
using FloatColorMatrix = float[MAX_COLORS][MAX_COLORS];
using IntColorVector = int[MAX_COLORS];
using IntColorMatrix = int[MAX_COLORS][MAX_COLORS];
using BoolColorMatrix = bool[MAX_COLORS][MAX_COLORS];

template <typename T>
struct Infinity
{
    static auto constexpr value = std::numeric_limits<T>::max();
};

struct SimulationParametersSpotValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    FloatColorVector radiationCellAgeStrength = {0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f};
    float cellMaxForce = 0.8f;
    FloatColorVector cellMinEnergy = {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f};
    float cellFusionVelocity = 0.4f;
    float cellMaxBindingEnergy = 500000.0f;
    IntColorVector cellColorTransitionDuration = {0, 0, 0, 0, 0, 0, 0};
    IntColorVector cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};
    FloatColorVector cellFunctionAttackerEnergyCost = {0, 0, 0, 0, 0, 0, 0};
    FloatColorMatrix cellFunctionAttackerFoodChainColorMatrix = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}};

    FloatColorVector cellFunctionAttackerGeometryDeviationExponent = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionAttackerConnectionsMismatchPenalty = {0, 0, 0, 0, 0, 0, 0};

    FloatColorVector cellFunctionConstructorMutationNeuronDataProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationDataProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationCellFunctionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationInsertionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationDeletionProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationTranslationProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationDuplicationProbability = {0, 0, 0, 0, 0, 0, 0};
    FloatColorVector cellFunctionConstructorMutationColorProbability = {0, 0, 0, 0, 0, 0, 0};

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
            if (cellFunctionConstructorMutationColorProbability[i] != other.cellFunctionConstructorMutationColorProbability[i]) {
                return false;
            }
        }
        return friction == other.friction && rigidity == other.rigidity && cellMaxForce == other.cellMaxForce
            && cellFusionVelocity == other.cellFusionVelocity
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
        ;
    }
};
