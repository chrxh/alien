#include "EngineInterfaceSettings.h"

#include "SimulationParameters.h"

SimulationParameters EngineInterfaceSettings::getDefaultSimulationParameters()
{
	SimulationParameters parameters;
	parameters.cellMinDistance = 0.3f;
	parameters.cellMaxDistance = 1.3f;
	parameters.cellMaxForce = 0.8f;
	parameters.cellMaxForceDecayProb = 0.2f;
    parameters.cellMinTokenUsages = 40000;
    parameters.cellTokenUsageDecayProb = 0.000001f;
    parameters.cellMaxBonds = 6;
	parameters.cellMaxToken = 3;
	parameters.cellMaxTokenBranchNumber = 6;
	parameters.cellCreationMaxConnection = 4;
	parameters.cellCreationTokenAccessNumber = 0;
	parameters.cellMinEnergy = 50.0f;
	parameters.cellTransformationProb = 0.2f;
	parameters.cellFusionVelocity = 0.4f;

	parameters.cellFunctionWeaponStrength = 0.1f;
    parameters.cellFunctionWeaponEnergyCost = 0.2f;
    parameters.cellFunctionWeaponGeometryDeviationExponent = 0;
    parameters.cellFunctionWeaponInhomogeneousColorFactor = 1.0f;
    parameters.cellFunctionComputerMaxInstructions = 15;
	parameters.cellFunctionComputerCellMemorySize = 8;
    parameters.cellFunctionConstructorOffspringCellEnergy = 100.0f;
	parameters.cellFunctionConstructorOffspringCellDistance = 1.6f;
    parameters.cellFunctionConstructorOffspringTokenEnergy = 60.0f;
    parameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy = false;
    parameters.cellFunctionConstructorTokenDataMutationProb = 0.002f;
    parameters.cellFunctionConstructorCellDataMutationProb = 0.002f;
    parameters.cellFunctionConstructorCellPropertyMutationProb = 0.002f;
    parameters.cellFunctionConstructorCellStructureMutationProb = 0.002f;
    parameters.cellFunctionSensorRange = 50.0f;
	parameters.cellFunctionCommunicatorRange = 50.0f;

	parameters.tokenMemorySize = 256;
	parameters.tokenMinEnergy = 3.0f;

	parameters.radiationExponent = 1;
	parameters.radiationFactor = 0.0002f;
	parameters.radiationProb = 0.03f;
	parameters.radiationVelocityMultiplier = 1.0f;
	parameters.radiationVelocityPerturbation = 0.5f;

	return parameters;
}
