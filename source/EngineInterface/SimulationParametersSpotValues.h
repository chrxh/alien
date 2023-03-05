#pragma once

#include "Constants.h"

//NOTE: header is also included in CUDA code
struct SimulationParametersSpotValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    float radiationFactor = 0.0002f;
    float cellMaxForce = 0.8f;
    float cellMinEnergy = 50.0f;
    float cellFusionVelocity = 0.4f;
    float cellMaxBindingEnergy = 500000.0f;
    int cellColorTransitionDuration[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    int cellColorTransitionTargetColor[MAX_COLORS] = {0, 1, 2, 3, 4, 5, 6};
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

    bool cellFunctionConstructorMutationNeuronDataProbabilityColorDependence = false;
    float cellFunctionConstructorMutationNeuronDataProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationDataProbabilityColorDependence = false;
    float cellFunctionConstructorMutationDataProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationCellFunctionProbabilityColorDependence = false;
    float cellFunctionConstructorMutationCellFunctionProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationInsertionProbabilityColorDependence = false;
    float cellFunctionConstructorMutationInsertionProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationDeletionProbabilityColorDependence = false;
    float cellFunctionConstructorMutationDeletionProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationTranslationProbabilityColorDependence = false;
    float cellFunctionConstructorMutationTranslationProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    bool cellFunctionConstructorMutationDuplicationProbabilityColorDependence = false;
    float cellFunctionConstructorMutationDuplicationProbability[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};

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
        return friction == other.friction && rigidity == other.rigidity && radiationFactor == other.radiationFactor && cellMaxForce == other.cellMaxForce
            && cellMinEnergy == other.cellMinEnergy && cellFusionVelocity == other.cellFusionVelocity
            && cellFunctionAttackerEnergyCost == other.cellFunctionAttackerEnergyCost
            && cellFunctionAttackerGeometryDeviationExponent == other.cellFunctionAttackerGeometryDeviationExponent
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
            && cellFunctionAttackerConnectionsMismatchPenalty == other.cellFunctionAttackerConnectionsMismatchPenalty
            && cellFunctionConstructorMutationNeuronDataProbabilityColorDependence == other.cellFunctionConstructorMutationNeuronDataProbabilityColorDependence
            && cellFunctionConstructorMutationDataProbabilityColorDependence == other.cellFunctionConstructorMutationDataProbabilityColorDependence
            && cellFunctionConstructorMutationCellFunctionProbabilityColorDependence == other.cellFunctionConstructorMutationCellFunctionProbabilityColorDependence
            && cellFunctionConstructorMutationInsertionProbabilityColorDependence == other.cellFunctionConstructorMutationInsertionProbabilityColorDependence
            && cellFunctionConstructorMutationDeletionProbabilityColorDependence == other.cellFunctionConstructorMutationDeletionProbabilityColorDependence
            && cellFunctionConstructorMutationTranslationProbabilityColorDependence == other.cellFunctionConstructorMutationTranslationProbabilityColorDependence
            && cellFunctionConstructorMutationDuplicationProbabilityColorDependence == other.cellFunctionConstructorMutationDuplicationProbabilityColorDependence
        ;
    }
};
