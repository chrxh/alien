#ifndef SIMULATIONACCESSAPI_H
#define SIMULATIONACCESSAPI_H

#include "model/Definitions.h"
#include "model/AccessPorts/Descriptions.h"

class SimulationAccessApi
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccessApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccessApi() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void addCell(CellDescription desc) = 0;

	//virtual void addData(DataDescription const &desc) = 0;
	//virtual void removeData(DataLightDescription &desc) = 0;

	//virtual void requestData(IntRect rect) = 0;
	//Q_SIGNAL void dataReadyToRetrieve();
	//virtual DataDescription& retrieveData() = 0;
	//virtual DataLightDescription& retrieveData() = 0;
};

#endif // SIMULATIONACCESSAPI_H
