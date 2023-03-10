#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersSpotActivatedValues
{
    bool friction = false;
    bool rigidity = false;
    bool radiationCellAgeStrength = false;
    bool cellMaxForce = false;
    bool cellMinEnergy = false;
    bool cellFusionVelocity = false;
    bool cellMaxBindingEnergy = false;
    bool cellColorTransition = false;
    bool cellFunctionAttackerEnergyCost = false;
    bool cellFunctionAttackerFoodChainColorMatrix = false;
    bool cellFunctionAttackerGeometryDeviationExponent = false;
    bool cellFunctionAttackerConnectionsMismatchPenalty = false;

    bool cellFunctionConstructorMutationNeuronDataProbability = false;
    bool cellFunctionConstructorMutationDataProbability = false;
    bool cellFunctionConstructorMutationCellFunctionProbability = false;
    bool cellFunctionConstructorMutationInsertionProbability = false;
    bool cellFunctionConstructorMutationDeletionProbability = false;
    bool cellFunctionConstructorMutationTranslationProbability = false;
    bool cellFunctionConstructorMutationDuplicationProbability = false;
    bool cellFunctionConstructorMutationColorProbability = false;

    bool operator==(SimulationParametersSpotActivatedValues const& other) const
    {
        return friction == other.friction && rigidity == other.rigidity && radiationCellAgeStrength == other.radiationCellAgeStrength && cellMaxForce == other.cellMaxForce
            && cellMinEnergy == other.cellMinEnergy && cellFusionVelocity == other.cellFusionVelocity
            && cellFunctionAttackerEnergyCost == other.cellFunctionAttackerEnergyCost && cellColorTransition == other.cellColorTransition
            && cellFunctionAttackerFoodChainColorMatrix == other.cellFunctionAttackerFoodChainColorMatrix
            && cellFunctionAttackerGeometryDeviationExponent == other.cellFunctionAttackerGeometryDeviationExponent
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
            && cellFunctionAttackerConnectionsMismatchPenalty == other.cellFunctionAttackerConnectionsMismatchPenalty
            && cellFunctionConstructorMutationNeuronDataProbability == other.cellFunctionConstructorMutationNeuronDataProbability
            && cellFunctionConstructorMutationDataProbability == other.cellFunctionConstructorMutationDataProbability
            && cellFunctionConstructorMutationCellFunctionProbability == other.cellFunctionConstructorMutationCellFunctionProbability
            && cellFunctionConstructorMutationInsertionProbability == other.cellFunctionConstructorMutationInsertionProbability
            && cellFunctionConstructorMutationDeletionProbability == other.cellFunctionConstructorMutationDeletionProbability
            && cellFunctionConstructorMutationTranslationProbability == other.cellFunctionConstructorMutationTranslationProbability
            && cellFunctionConstructorMutationDuplicationProbability == other.cellFunctionConstructorMutationDuplicationProbability
            && cellFunctionConstructorMutationColorProbability == other.cellFunctionConstructorMutationColorProbability
        ;
    }
};
