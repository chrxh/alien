#ifndef SIMULATIONACCESSSLOTWRAPPER_H
#define SIMULATIONACCESSSLOTWRAPPER_H

#include "model/context/UnitThreadController.h"
#include "model/context/SimulationContext.h"

class SimulationAccessSlotWrapper
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccessSlotWrapper(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccessSlotWrapper() = default;

	void init(SimulationContext* context);

	Q_SLOT void timestepCalculated();

protected:
	virtual void accessToSimulation() = 0;
};


#endif // SIMULATIONACCESSSLOTWRAPPER_H
