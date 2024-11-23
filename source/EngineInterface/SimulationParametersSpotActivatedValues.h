#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersSpotActivatedValues
{
    bool friction = false;
    bool rigidity = false;
    bool radiationAbsorption = false;
    bool radiationAbsorptionLowVelocityPenalty = false;
    bool radiationAbsorptionLowGenomeComplexityPenalty = false;
    bool radiationCellAgeStrength = false;
    bool radiationDisableSources = false;
    bool cellMaxForce = false;
    bool cellMinEnergy = false;
    bool cellDeathProbability = false;
    bool cellFusionVelocity = false;
    bool cellMaxBindingEnergy = false;
    bool cellInactiveMaxAge = false;
    bool cellColorTransition = false;
    bool cellFunctionAttackerEnergyCost = false;
    bool cellFunctionAttackerFoodChainColorMatrix = false;
    bool cellFunctionAttackerGenomeComplexityBonus = false;
    bool cellFunctionAttackerNewComplexMutantPenalty = false;
    bool cellFunctionAttackerGeometryDeviationExponent = false;
    bool cellFunctionAttackerConnectionsMismatchPenalty = false;

    bool cellCopyMutationNeuronData = false;
    bool cellCopyMutationCellProperties = false;
    bool cellCopyMutationCellFunction = false;
    bool cellCopyMutationGeometry = false;
    bool cellCopyMutationCustomGeometry = false; 
    bool cellCopyMutationInsertion = false;
    bool cellCopyMutationDeletion = false;
    bool cellCopyMutationTranslation = false;
    bool cellCopyMutationDuplication = false;
    bool cellCopyMutationCellColor = false;
    bool cellCopyMutationSubgenomeColor = false;
    bool cellCopyMutationGenomeColor = false;

    bool operator==(SimulationParametersSpotActivatedValues const& other) const
    {
        return friction == other.friction && rigidity == other.rigidity && radiationCellAgeStrength == other.radiationCellAgeStrength && cellMaxForce == other.cellMaxForce
            && cellMinEnergy == other.cellMinEnergy && cellFusionVelocity == other.cellFusionVelocity
            && cellFunctionAttackerEnergyCost == other.cellFunctionAttackerEnergyCost && cellColorTransition == other.cellColorTransition
            && cellFunctionAttackerFoodChainColorMatrix == other.cellFunctionAttackerFoodChainColorMatrix
            && cellFunctionAttackerGeometryDeviationExponent == other.cellFunctionAttackerGeometryDeviationExponent
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
            && cellFunctionAttackerConnectionsMismatchPenalty == other.cellFunctionAttackerConnectionsMismatchPenalty
            && cellCopyMutationNeuronData == other.cellCopyMutationNeuronData
            && cellCopyMutationCellProperties == other.cellCopyMutationCellProperties
            && cellCopyMutationCellFunction == other.cellCopyMutationCellFunction
            && cellCopyMutationInsertion == other.cellCopyMutationInsertion
            && cellCopyMutationDeletion == other.cellCopyMutationDeletion
            && cellCopyMutationTranslation == other.cellCopyMutationTranslation
            && cellCopyMutationDuplication == other.cellCopyMutationDuplication
            && cellCopyMutationSubgenomeColor == other.cellCopyMutationSubgenomeColor
            && radiationAbsorption == other.radiationAbsorption
            && cellCopyMutationCustomGeometry == other.cellCopyMutationCustomGeometry
            && cellCopyMutationGeometry == other.cellCopyMutationGeometry
            && cellCopyMutationGenomeColor == other.cellCopyMutationGenomeColor
            && cellCopyMutationCellColor == other.cellCopyMutationCellColor
            && radiationAbsorptionLowGenomeComplexityPenalty == other.radiationAbsorptionLowGenomeComplexityPenalty
            && cellFunctionAttackerGenomeComplexityBonus == other.cellFunctionAttackerGenomeComplexityBonus
            && radiationAbsorptionLowVelocityPenalty == other.radiationAbsorptionLowVelocityPenalty
            && cellFunctionAttackerNewComplexMutantPenalty == other.cellFunctionAttackerNewComplexMutantPenalty
            && cellInactiveMaxAge == other.cellInactiveMaxAge && radiationDisableSources == other.radiationDisableSources
            && cellDeathProbability == other.cellDeathProbability
        ;
    }
};
