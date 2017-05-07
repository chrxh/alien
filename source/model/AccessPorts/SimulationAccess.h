#ifndef SIMULATIONACCESS_H
#define SIMULATIONACCESS_H

#include "model/Definitions.h"
#include "model/SimulationAccessApi.h"

class SimulationAccess
	: public SimulationAccessApi
{
	Q_OBJECT
public:
	SimulationAccess(QObject* parent = nullptr) : SimulationAccessApi(parent) {}
	virtual ~SimulationAccess() = default;
};

#endif // SIMULATIONACCESS_H
