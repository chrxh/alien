#pragma once

#include <QObject>

#include "ModelBasic/Descriptions.h"
#include "Definitions.h"

class VersionController
	: public QObject
{
	Q_OBJECT

public:
	VersionController(QObject * parent = nullptr);
	virtual ~VersionController() = default;

	virtual void init(SimulationContext* context, SimulationAccess* access);

	virtual bool isStackEmpty();
	virtual void clearStack();
	virtual void loadSimulationContentFromStack();
	virtual void saveSimulationContentToStack();

	virtual void makeSnapshot();
	virtual void restoreSnapshot();

private:
	Q_SLOT void dataReadyToRetrieve();

	IntVector2D _universeSize;
	SimulationAccess* _access = nullptr;

	enum class TargetForReceivedData { Stack, Snapshot};
	optional<TargetForReceivedData> _target;
	list<DataDescription> _stack;
	optional<DataDescription> _snapshot;
};
