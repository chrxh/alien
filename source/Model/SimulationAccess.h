#pragma once

#include "Model/Definitions.h"
#include "Model/Descriptions.h"

class MODEL_EXPORT SimulationAccess
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccess(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccess() = default;

	virtual void init(SimulationContext* context) = 0;

	virtual void updateData(DataChangeDescription const &desc) = 0;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) = 0;
	virtual void requireImage(IntRect rect, QImage* target) = 0;

	Q_SIGNAL void dataReadyToRetrieve();
	Q_SIGNAL void imageReady();
	virtual DataDescription const& retrieveData() = 0;
};

