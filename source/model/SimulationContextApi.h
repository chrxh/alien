#ifndef SIMULATIONCONTEXTAPI_H
#define SIMULATIONCONTEXTAPI_H

#include "model/Definitions.h"

class SimulationContextApi
	: public QObject
{
Q_OBJECT
public:
	SimulationContextApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContextApi() = default;
};

#endif // SIMULATIONCONTEXTAPI_H
