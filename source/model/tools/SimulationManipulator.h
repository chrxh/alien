#ifndef MAPMANIPULATOR_H
#define MAPMANIPULATOR_H

#include "model/Definitions.h"
#include "model/SimulationManipulatorApi.h"

class SimulationManipulator
	: public SimulationManipulatorApi
{
	Q_OBJECT
public:
	SimulationManipulator(QObject* parent = nullptr) : SimulationManipulatorApi(parent) {}
	virtual ~SimulationManipulator() = default;
};

#endif // MAPMANIPULATOR_H
