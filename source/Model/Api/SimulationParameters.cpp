#include "SimulationParameters.h"


SimulationParameters::SimulationParameters(QObject * parent)
	: QObject(parent)
{
}

SimulationParameters * SimulationParameters::clone(QObject * parent) const
{
	auto parameters = new SimulationParameters(parent);
	parameters->clusterMaxRadius = clusterMaxRadius;
	parameters->cellMutationProb = cellMutationProb;
	parameters->cellMinDistance = cellMinDistance;
	parameters->cellMaxDistance = cellMaxDistance;
	parameters->cellMass_Reciprocal = cellMass_Reciprocal;
	parameters->callMaxForce = callMaxForce;
	parameters->cellMaxForceDecayProb = cellMaxForceDecayProb;
	parameters->cellMaxBonds = cellMaxBonds;
	parameters->cellMaxToken = cellMaxToken;
	parameters->cellMaxTokenBranchNumber = cellMaxTokenBranchNumber;
	parameters->cellFunctionConstructorOffspringCellEnergy = cellFunctionConstructorOffspringCellEnergy;
	parameters->cellCreationMaxConnection = cellCreationMaxConnection;
	parameters->cellCreationTokenAccessNumber = cellCreationTokenAccessNumber;
	parameters->cellMinEnergy = cellMinEnergy;
	parameters->cellTransformationProb = cellTransformationProb;
	parameters->cellFusionVelocity = cellFusionVelocity;

	parameters->cellFunctionWeaponStrength = cellFunctionWeaponStrength;
	parameters->cellFunctionComputerMaxInstructions = cellFunctionComputerMaxInstructions;
	parameters->cellFunctionComputerCellMemorySize = cellFunctionComputerCellMemorySize;
	parameters->cellFunctionConstructorOffspringCellDistance = cellFunctionConstructorOffspringCellDistance;
	parameters->cellFunctionSensorRange = cellFunctionSensorRange;
	parameters->cellFunctionCommunicatorRange = cellFunctionCommunicatorRange;

	parameters->tokenMemorySize = tokenMemorySize;
	parameters->cellFunctionConstructorOffspringTokenEnergy = cellFunctionConstructorOffspringTokenEnergy;
	parameters->tokenMinEnergy = tokenMinEnergy;

	parameters->radiationExponent = radiationExponent;
	parameters->radiationFactor = radiationFactor;
	parameters->radiationProb = radiationProb;

	parameters->radiationVelocityMultiplier = radiationVelocityMultiplier;
	parameters->radiationVelocityPerturbation = radiationVelocityPerturbation;
	return parameters;
}
