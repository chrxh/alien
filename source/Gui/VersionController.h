#pragma once

#include <QObject>

#include "Definitions.h"

class VersionController
	: public QObject
{
	Q_OBJECT

public:
	VersionController(QObject * parent = nullptr);
	virtual ~VersionController() = default;

	virtual void init(SimulationContext* context);

	virtual bool isStackEmpty();
	virtual void clearStack();
	virtual void loadSimulationContentFromStack();
	virtual void saveSimulationContentToStack();

private:
	Q_SLOT void dataReadyToRetrieve();

	SimulationAccess* _access = nullptr;

	IntVector2D _universeSize;
	list<DataDescription> _stack;
};
