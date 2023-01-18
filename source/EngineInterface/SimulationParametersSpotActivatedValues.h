#pragma once


struct SimulationParametersSpotActivatedValues
{
    bool friction = false;
    bool rigidity = false;
    bool radiationFactor = false;
    bool cellMaxForce = false;
    bool cellMinEnergy = false;
    bool cellFusionVelocity = false;
    bool cellMaxBindingEnergy = false;
    bool cellColorTransition = false;
    bool cellFunctionAttackerEnergyCost = false;
    bool cellFunctionAttackerFoodChainColorMatrix = false;

    bool operator==(SimulationParametersSpotActivatedValues const& other) const
    {
        return friction == other.friction && rigidity == other.rigidity
            && radiationFactor == other.radiationFactor && cellMaxForce == other.cellMaxForce
            && cellMinEnergy == other.cellMinEnergy && cellFusionVelocity == other.cellFusionVelocity
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy && cellColorTransition == other.cellColorTransition
            && cellFunctionAttackerEnergyCost == other.cellFunctionAttackerEnergyCost
            && cellFunctionAttackerFoodChainColorMatrix == other.cellFunctionAttackerFoodChainColorMatrix;
    }
};
