#include "SimulationParameters.h"


SimulationParameters::SimulationParameters(QObject * parent)
	: QObject(parent)
{
}

SimulationParameters * SimulationParameters::clone(QObject * parent) const
{
	auto parameters = new SimulationParameters(parent);
	parameters->cellMutationProb = cellMutationProb;
	parameters->cellMinDistance = cellMinDistance;
	parameters->cellMaxDistance = cellMaxDistance;
	parameters->cellMass_Reciprocal = cellMass_Reciprocal;
	parameters->callMaxForce = callMaxForce;
	parameters->cellMaxForceDecayProb = cellMaxForceDecayProb;
	parameters->cellMaxBonds = cellMaxBonds;
	parameters->cellMaxToken = cellMaxToken;
	parameters->cellMaxTokenBranchNumber = cellMaxTokenBranchNumber;
	parameters->cellCreationEnergy = cellCreationEnergy;
	parameters->cellCreationMaxConnection = cellCreationMaxConnection;
	parameters->cellCreationTokenAccessNumber = cellCreationTokenAccessNumber;
	parameters->cellMinEnergy = cellMinEnergy;
	parameters->cellTransformationProb = cellTransformationProb;
	parameters->cellFusionVelocity = cellFusionVelocity;

	parameters->cellFunctionWeaponStrength = cellFunctionWeaponStrength;
	parameters->cellFunctionComputerMaxInstructions = cellFunctionComputerMaxInstructions;
	parameters->cellFunctionComputerCellMemorySize = cellFunctionComputerCellMemorySize;
	parameters->cellFunctionConstructorOffspringDistance = cellFunctionConstructorOffspringDistance;
	parameters->cellFunctionSensorRange = cellFunctionSensorRange;
	parameters->cellFunctionCommunicatorRange = cellFunctionCommunicatorRange;

	parameters->tokenMemorySize = tokenMemorySize;
	parameters->tokenCreationEnergy = tokenCreationEnergy;
	parameters->tokenMinEnergy = tokenMinEnergy;

	parameters->radiationExponent = radiationExponent;
	parameters->radiationFactor = radiationFactor;
	parameters->radiationProb = radiationProb;

	parameters->radiationVelocityMultiplier = radiationVelocityMultiplier;
	parameters->radiationVelocityPerturbation = radiationVelocityPerturbation;
	return parameters;
}

void SimulationParameters::serializePrimitives(QDataStream & stream)
{
	stream << cellMinDistance;
	stream << cellMaxDistance;
	stream << cellMass_Reciprocal;
	stream << callMaxForce;
	stream << cellMaxForceDecayProb;
	stream << cellMaxBonds;
	stream << cellMaxToken;
	stream << cellMaxTokenBranchNumber;
	stream << cellCreationEnergy;
	stream << cellCreationMaxConnection;
	stream << cellCreationTokenAccessNumber;
	stream << cellMinEnergy;
	stream << cellTransformationProb;
	stream << cellFusionVelocity;
	stream << cellFunctionWeaponStrength;
	stream << cellFunctionComputerMaxInstructions;
	stream << cellFunctionComputerCellMemorySize;
	stream << tokenMemorySize;
	stream << cellFunctionConstructorOffspringDistance;
	stream << cellFunctionSensorRange;
	stream << cellFunctionCommunicatorRange;
	stream << tokenCreationEnergy;
	stream << tokenMinEnergy;
	stream << radiationExponent;
	stream << radiationFactor;
	stream << radiationProb;
	stream << radiationVelocityMultiplier;
	stream << radiationVelocityPerturbation;
}

void SimulationParameters::deserializePrimitives(QDataStream & stream)
{
	stream >> cellMinDistance;
	stream >> cellMaxDistance;
	stream >> cellMass_Reciprocal;
	stream >> callMaxForce;
	stream >> cellMaxForceDecayProb;
	stream >> cellMaxBonds;
	stream >> cellMaxToken;
	stream >> cellMaxTokenBranchNumber;
	stream >> cellCreationEnergy;
	stream >> cellCreationMaxConnection;
	stream >> cellCreationTokenAccessNumber;
	stream >> cellMinEnergy;
	stream >> cellTransformationProb;
	stream >> cellFusionVelocity;
	stream >> cellFunctionWeaponStrength;
	stream >> cellFunctionComputerMaxInstructions;
	stream >> cellFunctionComputerCellMemorySize;
	stream >> tokenMemorySize;
	stream >> cellFunctionConstructorOffspringDistance;
	stream >> cellFunctionSensorRange;
	stream >> cellFunctionCommunicatorRange;
	stream >> tokenCreationEnergy;
	stream >> tokenMinEnergy;
	stream >> radiationExponent;
	stream >> radiationFactor;
	stream >> radiationProb;
	stream >> radiationVelocityMultiplier;
	stream >> radiationVelocityPerturbation;
}
