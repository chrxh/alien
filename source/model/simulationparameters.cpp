#include "simulationparameters.h"


void SimulationParameters::setParameters(SimulationParameters * other)
{
	cellMutationProb = other->cellMutationProb;
	cellMinDistance = other->cellMinDistance;
	cellMaxDistance = other->cellMaxDistance;
	cellMass_Reciprocal = other->cellMass_Reciprocal;
	callMaxForce = other->callMaxForce;
	cellMaxForceDecayProb = other->cellMaxForceDecayProb;
	cellMaxBonds = other->cellMaxBonds;
	cellMaxToken = other->cellMaxToken;
	cellMaxTokenBranchNumber = other->cellMaxTokenBranchNumber;
	cellCreationEnergy = other->cellCreationEnergy;
	NEW_CELL_MAX_CONNECTION = other->NEW_CELL_MAX_CONNECTION;
	NEW_CELL_TOKEN_ACCESS_NUMBER = other->NEW_CELL_TOKEN_ACCESS_NUMBER;
	cellMinEnergy = other->cellMinEnergy;
	cellTransformationProb = other->cellTransformationProb;
	cellFusionVelocity = other->cellFusionVelocity;

	cellFunctionWeaponStrength = other->cellFunctionWeaponStrength;
	cellFunctionComputerMaxInstructions = other->cellFunctionComputerMaxInstructions;
	cellFunctionComputerCellMemorySize = other->cellFunctionComputerCellMemorySize;
	cellFunctionComputerTokenMemorySize = other->cellFunctionComputerTokenMemorySize;
	cellFunctionConstructorOffspringDistance = other->cellFunctionConstructorOffspringDistance;
	cellFunctionSensorRange = other->cellFunctionSensorRange;
	cellFunctionCommunicatorRange = other->cellFunctionCommunicatorRange;

	tokenCreationEnergy = other->tokenCreationEnergy;
	tokenMinEnergy = other->tokenMinEnergy;

	radiationExponent = other->radiationExponent;
	radiationFactor = other->radiationFactor;
	radiationProb = other->radiationProb;

	radiationVelocityMultiplier = other->radiationVelocityMultiplier;
	radiationVelocityPerturbation = other->radiationVelocityPerturbation;
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
	stream << NEW_CELL_MAX_CONNECTION;
	stream << NEW_CELL_TOKEN_ACCESS_NUMBER;
	stream << cellMinEnergy;
	stream << cellTransformationProb;
	stream << cellFusionVelocity;
	stream << cellFunctionWeaponStrength;
	stream << cellFunctionComputerMaxInstructions;
	stream << cellFunctionComputerCellMemorySize;
	stream << cellFunctionComputerTokenMemorySize;
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
	stream >> NEW_CELL_MAX_CONNECTION;
	stream >> NEW_CELL_TOKEN_ACCESS_NUMBER;
	stream >> cellMinEnergy;
	stream >> cellTransformationProb;
	stream >> cellFusionVelocity;
	stream >> cellFunctionWeaponStrength;
	stream >> cellFunctionComputerMaxInstructions;
	stream >> cellFunctionComputerCellMemorySize;
	stream >> cellFunctionComputerTokenMemorySize;
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
