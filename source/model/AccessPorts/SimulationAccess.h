#ifndef SIMULATIONACCESS_H
#define SIMULATIONACCESS_H

#include "model/Definitions.h"
#include "model/Entities/Descriptions.h"

class SimulationAccess
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccess(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccess() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void updateData(DataDescription const &desc) = 0;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) = 0;
	Q_SIGNAL void dataReadyToRetrieve();
	virtual DataDescription const& retrieveData() = 0;
};

#endif // SIMULATIONACCESS_H
