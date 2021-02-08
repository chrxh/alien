#pragma once

#include <mutex>

#include "Definitions.h"
#include "Descriptions.h"

class ENGINEINTERFACE_EXPORT SimulationAccess
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
    virtual void requirePixelImage(IntRect rect, QImagePtr const& target, std::mutex& mutex) = 0;
    virtual void requireVectorImage(IntRect rect, double zoom, QImagePtr const& target, std::mutex& mutex) = 0;
    virtual void selectEntities(IntVector2D const& pos) = 0;
    virtual void deselectAll() = 0;
    virtual void applyAction(PhysicalAction const& action) = 0;


	Q_SIGNAL void dataReadyToRetrieve();
	Q_SIGNAL void dataUpdated();
	Q_SIGNAL void imageReady();
	virtual DataDescription const& retrieveData() = 0;
};

