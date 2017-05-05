#ifndef SIMULATIONACCESSAPI_H
#define SIMULATIONACCESSAPI_H

#include "model/Definitions.h"
#include "model/tools/CellDescription.h"

class SimulationAccessApi
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccessApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccessApi() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void addCell(CellDescription desc) = 0;

	virtual std::vector<UnitContextApi*> getAndLockData(IntRect rect) = 0;
	virtual void unlock() = 0;
};

#endif // SIMULATIONACCESSAPI_H
