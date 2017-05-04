#ifndef SIMULATIONCONTEXTWRAPPER_H
#define SIMULATIONCONTEXTWRAPPER_H

#include "model/Definitions.h"

class SimulationContextWrapper
	: public QObject
{
	Q_OBJECT
public:
	SimulationContextWrapper(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContextWrapper() = default;
};

#endif // SIMULATIONCONTEXTWRAPPER_H
