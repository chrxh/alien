#ifndef SIMULATIONACCESS_H
#define SIMULATIONACCESS_H

#include "Model/Definitions.h"
#include "Model/Entities/Descriptions.h"

class MODEL_EXPORT SimulationAccess
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccess(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccess() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void updateData(DataChangeDescription const &desc) = 0;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) = 0;
	virtual void requireImage(IntRect rect, QImage* target) = 0;

	Q_SIGNAL void dataReadyToRetrieve();
	Q_SIGNAL void imageReady();
	virtual DataDescription const& retrieveData() = 0;
};

#endif // SIMULATIONACCESS_H
