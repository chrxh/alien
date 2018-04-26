#pragma once

#include "Model/Api/Definitions.h"

class MODEL_EXPORT SimulationParameters
	: public QObject
{
	Q_OBJECT
public:
	SimulationParameters(QObject* parent = nullptr);
	virtual ~SimulationParameters() = default;

	SimulationParameters* clone(QObject* parent = nullptr) const;

	qreal clusterMaxRadius = 0.0;

	qreal cellMutationProb = 0.0;
	qreal cellMinDistance = 0.0;
	qreal cellMaxDistance = 0.0;
	qreal cellMass_Reciprocal = 0.0; //related to 1/mass
	qreal callMaxForce = 0.0;
	qreal cellMaxForceDecayProb = 0.0;
	int cellMaxBonds = 0;
	int cellMaxToken = 0;
	int cellMaxTokenBranchNumber = 0;
	int cellCreationMaxConnection = 0;	//TODO: add to editor
	int cellCreationTokenAccessNumber = 0; //TODO: add to editor
	qreal cellMinEnergy = 0.0;
	qreal cellTransformationProb = 0.0;
	qreal cellFusionVelocity = 0.0;

	int cellFunctionComputerMaxInstructions = 0;
	int cellFunctionComputerCellMemorySize = 0;
	qreal cellFunctionWeaponStrength = 0.0;
	qreal cellFunctionConstructorOffspringCellEnergy = 0.0;
	qreal cellFunctionConstructorOffspringCellDistance = 0.0;
	qreal cellFunctionConstructorOffspringTokenEnergy = 0.0;
	qreal cellFunctionSensorRange = 0.0;
	qreal cellFunctionCommunicatorRange = 0.0;

	int tokenMemorySize = 0;
	qreal tokenMinEnergy = 0.0;

	qreal radiationExponent = 0.0;
	qreal radiationFactor = 0.0;
	qreal radiationProb = 0.0;

	qreal radiationVelocityMultiplier = 0.0;
	qreal radiationVelocityPerturbation = 0.0;
};

