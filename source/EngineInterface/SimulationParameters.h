#pragma once

struct SimulationParameters
{
    float cellMinDistance = 0.3f;
    float cellMaxDistance = 1.3f;
    float cellMaxForce = 0.8f;
    float cellMaxForceDecayProb = 0.2f;
    int cellMinTokenUsages = 40000;
    float cellTokenUsageDecayProb = 0.000001f;
    int cellMaxBonds = 6;
    int cellMaxToken = 3;
    int cellMaxTokenBranchNumber = 6;
    int cellCreationMaxConnection = 4;
    int cellCreationTokenAccessNumber = 0;
    float cellMinEnergy = 50.0f;
    float cellTransformationProb = 0.2f;
    float cellFusionVelocity = 0.4f;

    float cellFunctionWeaponStrength = 0.1f;
    float cellFunctionWeaponEnergyCost = 0.2f;
    float cellFunctionWeaponGeometryDeviationExponent = 0;
    float cellFunctionWeaponInhomogeneousColorFactor = 1.0f;
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
    float cellFunctionSensorRange = 50.0f;
    float cellFunctionCommunicatorRange = 50.0f;

    int tokenMemorySize = 256;
    float tokenMinEnergy = 3.0f;

    float radiationExponent = 1;
    float radiationFactor = 0.0002f;
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
};

