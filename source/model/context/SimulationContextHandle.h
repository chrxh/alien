#ifndef SIMULATIONCONTEXTHANDLE_H
#define SIMULATIONCONTEXTHANDLE_H

#include "model/Definitions.h"

class SimulationContextHandle
	: public QObject
{
	Q_OBJECT
public:
	SimulationContextHandle(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContextHandle() = default;
};

#endif // SIMULATIONCONTEXTHANDLE_H
