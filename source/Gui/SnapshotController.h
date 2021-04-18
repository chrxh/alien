#pragma once

#include <QObject>

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class SnapshotController
	: public QObject
{
	Q_OBJECT

public:
	SnapshotController(QObject * parent = nullptr);
	virtual ~SnapshotController() = default;

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
    SimulationContext* _context = nullptr;
	SimulationAccess* _access = nullptr;

	enum class TargetForReceivedData { Stack, Snapshot};
	boost::optional<TargetForReceivedData> _target;
    struct SnapshotData
    {
        DataDescription data;
        int timestep;
    };
	list<SnapshotData> _stack;
	boost::optional<SnapshotData> _snapshot;
};
