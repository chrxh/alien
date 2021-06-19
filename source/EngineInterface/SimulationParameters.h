#pragma once

struct SimulationParameters
{
	float cellMinDistance = 0.0;
	float cellMaxDistance = 0.0;
	float cellMaxForce = 0.0;
	float cellMaxForceDecayProb = 0.0;
    int cellMinTokenUsages = 0;
    float cellTokenUsageDecayProb = 0;
    int cellMaxBonds = 0;
	int cellMaxToken = 0;
	int cellMaxTokenBranchNumber = 0;
	int cellCreationMaxConnection = 0;	//TODO: add to editor
	int cellCreationTokenAccessNumber = 0; //TODO: add to editor
    float cellMinEnergy = 0.0;
	float cellTransformationProb = 0.0;
	float cellFusionVelocity = 0.0;

	int cellFunctionComputerMaxInstructions = 0;
	int cellFunctionComputerCellMemorySize = 0;
	float cellFunctionWeaponStrength = 0.0;
    float cellFunctionWeaponEnergyCost = 0.0;
    float cellFunctionWeaponGeometryDeviationExponent = 0.0;
    float cellFunctionWeaponInhomogeneousColorFactor = 0.0;
    float cellFunctionConstructorOffspringCellEnergy = 0.0;
	float cellFunctionConstructorOffspringCellDistance = 0.0;
	float cellFunctionConstructorOffspringTokenEnergy = 0.0;
    bool cellFunctionConstructorOffspringTokenSuppressMemoryCopy = false;
    float cellFunctionConstructorTokenDataMutationProb = 0.0;
    float cellFunctionConstructorCellDataMutationProb = 0.0;
    float cellFunctionConstructorCellPropertyMutationProb = 0.0;
    float cellFunctionConstructorCellStructureMutationProb = 0.0;
    float cellFunctionSensorRange = 0.0;
	float cellFunctionCommunicatorRange = 0.0;

	int tokenMemorySize = 0;
	float tokenMinEnergy = 0.0;

	float radiationExponent = 0.0;
	float radiationFactor = 0.0;
	float radiationProb = 0.0;
	float radiationVelocityMultiplier = 0.0;
	float radiationVelocityPerturbation = 0.0;
};

