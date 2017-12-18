#include "Model/Api/ModelBuilderFacade.h"

#include "Base/ServiceLocator.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceProperties.h"

#include "VersionController.h"


VersionController::VersionController(QObject * parent) : QObject(parent)
{
	
}

void VersionController::init(SimulationContext* context)
{
	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto access = facade->buildSimulationAccess();
	SET_CHILD(_access, access);
	_access->init(context);
	_universeSize = context->getSpaceProperties()->getSize();

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
	_access->updateData(_stack.back());
	_stack.pop_back();
}

void VersionController::saveSimulationContentToStack()
{
	ResolveDescription resolveDesc;
	_access->requireData({ { 0, 0 }, _universeSize }, resolveDesc);
}

void VersionController::dataReadyToRetrieve()
{
	_stack.push_back(_access->retrieveData());
}
