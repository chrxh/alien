#include "EngineInterface/EngineInterfaceBuilderFacade.h"

#include "Base/ServiceLocator.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SpaceProperties.h"

#include "SnapshotController.h"


SnapshotController::SnapshotController(QObject * parent) : QObject(parent)
{
	
}

void SnapshotController::init(SimulationContext* context, SimulationAccess* access)
{
    _context = context;
	SET_CHILD(_access, access);
	_universeSize = context->getSpaceProperties()->getSize();
	_stack.clear();
	_snapshot.reset();

	connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &SnapshotController::dataReadyToRetrieve);
}

bool SnapshotController::isStackEmpty()
{
	return _stack.empty();
}

void SnapshotController::clearStack()
{
	_stack.clear();
}

void SnapshotController::loadSimulationContentFromStack()
{
	if (_stack.empty()) {
		return;
	}
	_access->clear();
	_access->updateData(_stack.back().data);
	_stack.pop_back();
}

void SnapshotController::saveSimulationContentToStack()
{
	_target = TargetForReceivedData::Stack;
	_access->requireData({ { 0, 0 }, _universeSize }, ResolveDescription());
}

void SnapshotController::makeSnapshot()
{
	_target = TargetForReceivedData::Snapshot;
	_access->requireData({ { 0, 0 }, _universeSize }, ResolveDescription());
}

void SnapshotController::restoreSnapshot()
{
	if (!_snapshot) {
		return;
	}
	_access->clear();
	_access->updateData(_snapshot->data);
    _context->setTimestep(_snapshot->timestep);
}

void SnapshotController::dataReadyToRetrieve()
{
	if (!_target) {
		return;
	}
    auto const timestep = _context->getTimestep();
	if (*_target == TargetForReceivedData::Stack) {
        _stack.emplace_back(SnapshotData{ _access->retrieveData(), timestep });
	}
	if (*_target == TargetForReceivedData::Snapshot) {
        _snapshot = SnapshotData{ _access->retrieveData(), timestep };
	}
	_target.reset();
}
