#pragma once

#include "Definitions.h"
#include "Descriptions.h"

class MODELBASIC_EXPORT SimulationAccess
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccess(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccess() = default;

	virtual void clear() = 0;
	virtual void updateData(DataChangeDescription const &desc) = 0;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) = 0;
    virtual void requireData(ResolveDescription const& resolveDesc) = 0;
    virtual void requireImage(IntRect rect, QImagePtr const& target) = 0;
    virtual void applyAction(PhysicalAction const& action) = 0;

	Q_SIGNAL void dataReadyToRetrieve();
	Q_SIGNAL void dataUpdated();
	Q_SIGNAL void imageReady();
	virtual DataDescription const& retrieveData() = 0;
};

