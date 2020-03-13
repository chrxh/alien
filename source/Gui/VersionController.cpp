#include "ModelBasic/ModelBasicBuilderFacade.h"

#include "Base/ServiceLocator.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"

#include "VersionController.h"


VersionController::VersionController(QObject * parent) : QObject(parent)
{
	
}

void VersionController::init(SimulationContext* context, SimulationAccess* access)
{
    _context = context;
	SET_CHILD(_access, access);
	_universeSize = context->getSpaceProperties()->getSize();
	_stack.clear();
	_snapshot.reset();

	connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &VersionController::dataReadyToRetrieve);
}

bool VersionController::isStackEmpty()
{
	return _stack.empty();
}

void VersionController::clearStack()
{
	_stack.clear();
}

void VersionController::loadSimulationContentFromStack()
{
	if (_stack.empty()) {
		return;
	}
	_access->clear();
	_access->updateData(_stack.back().data);
	_stack.pop_back();
}

void VersionController::saveSimulationContentToStack()
{
	_target = TargetForReceivedData::Stack;
	_access->requireData({ { 0, 0 }, _universeSize }, ResolveDescription());
}

void VersionController::makeSnapshot()
{
	_target = TargetForReceivedData::Snapshot;
	_access->requireData({ { 0, 0 }, _universeSize }, ResolveDescription());
}

void VersionController::restoreSnapshot()
{
	if (!_snapshot) {
		return;
	}
	_access->clear();
	_access->updateData(_snapshot->data);
    _context->setTimestep(_snapshot->timestep);
}

void VersionController::dataReadyToRetrieve()
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
