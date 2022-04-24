#pragma once

#include "SimulationParametersSpotValues.h"

struct SimulationParameters
{
    SimulationParametersSpotValues spotValues;

    float timestepSize = 1.0f;            //
    float cellMaxVel = 2.0f;              //
    float cellMaxBindingDistance = 2.6f;  //
    float cellRepulsionStrength = 0.08f;        //

    float cellMinDistance = 0.3f;           //
    float cellMaxCollisionDistance = 1.3f;  //
    float cellMaxForceDecayProb = 0.2f;
    int cellMaxBonds = 6;  //
    int cellMaxToken = 3;
    int cellMaxTokenBranchNumber = 6;
    int cellCreationMaxConnection = 4;
    int cellCreationTokenAccessNumber = 0;
    float cellTransformationProb = 0.2f;

    float cellFunctionWeaponStrength = 0.1f;
    int cellFunctionComputerMaxInstructions = 15;
    int cellFunctionComputerCellMemorySize = 8;
    float cellFunctionConstructorOffspringCellEnergy = 100.0f;
    float cellFunctionConstructorOffspringCellDistance = 1.6f;
    float cellFunctionConstructorOffspringTokenEnergy = 60.0f;
    bool cellFunctionConstructorOffspringTokenSuppressMemoryCopy = false;
    float cellFunctionConstructorTokenDataMutationProb = 0.002f;
    float cellFunctionConstructorCellDataMutationProb = 0.002f;
    float cellFunctionConstructorCellPropertyMutationProb = 0.002f;
    float cellFunctionConstructorCellStructureMutationProb = 0.002f;
    float cellFunctionSensorRange = 255.0f;
    float cellFunctionCommunicatorRange = 50.0f;

    int tokenMemorySize = 256;
    float tokenMinEnergy = 3.0f;

    float radiationExponent = 1;
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;

    bool operator==(SimulationParameters const& other) const
    {
        return spotValues == other.spotValues && timestepSize == other.timestepSize && cellMaxVel == other.cellMaxVel
            && cellMaxBindingDistance == other.cellMaxBindingDistance && cellMinDistance == other.cellMinDistance
            && cellMaxCollisionDistance == other.cellMaxCollisionDistance
            && cellMaxForceDecayProb == other.cellMaxForceDecayProb
            && cellMaxBonds == other.cellMaxBonds
            && cellMaxToken == other.cellMaxToken && cellMaxTokenBranchNumber == other.cellMaxTokenBranchNumber
            && cellCreationMaxConnection == other.cellCreationMaxConnection
            && cellCreationTokenAccessNumber == other.cellCreationTokenAccessNumber
            && cellTransformationProb == other.cellTransformationProb
            && cellFunctionWeaponStrength == other.cellFunctionWeaponStrength
            && cellFunctionComputerMaxInstructions == other.cellFunctionComputerMaxInstructions
            && cellFunctionComputerCellMemorySize == other.cellFunctionComputerCellMemorySize
            && cellFunctionConstructorOffspringCellEnergy == other.cellFunctionConstructorOffspringCellEnergy
            && cellFunctionConstructorOffspringCellDistance == other.cellFunctionConstructorOffspringCellDistance
            && cellFunctionConstructorOffspringTokenEnergy == other.cellFunctionConstructorOffspringTokenEnergy
            && cellFunctionConstructorOffspringTokenSuppressMemoryCopy
            == other.cellFunctionConstructorOffspringTokenSuppressMemoryCopy
            && cellFunctionConstructorTokenDataMutationProb == other.cellFunctionConstructorTokenDataMutationProb
            && cellFunctionConstructorCellDataMutationProb == other.cellFunctionConstructorCellDataMutationProb
            && cellFunctionConstructorCellPropertyMutationProb == other.cellFunctionConstructorCellPropertyMutationProb
            && cellFunctionConstructorCellStructureMutationProb
            == other.cellFunctionConstructorCellStructureMutationProb
            && cellFunctionSensorRange == other.cellFunctionSensorRange
            && cellFunctionCommunicatorRange == other.cellFunctionCommunicatorRange
            && tokenMemorySize == other.tokenMemorySize && tokenMinEnergy == other.tokenMinEnergy
            && radiationExponent == other.radiationExponent && radiationProb == other.radiationProb
            && radiationVelocityMultiplier == other.radiationVelocityMultiplier
            && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellRepulsionStrength == other.cellRepulsionStrength;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
