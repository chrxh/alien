#ifndef SIMULATIONBASICCONTEXT_H
#define SIMULATIONBASICCONTEXT_H

#include "model/Definitions.h"

class SimulationContextHandle
	: public QObject
{
	Q_OBJECT
public:
	SimulationContextHandle(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContextHandle() = default;
};

#endif // SIMULATIONBASICCONTEXT_H
