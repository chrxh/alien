#include "ModelInterface/SimulationParameters.h"
#include "UnitThreadController.h"
#include "UnitGrid.h"
#include "UnitContext.h"
#include "Unit.h"
#include "SimulationContextImpl.h"

#include "SimulationAttributeSetter.h"

SimulationAttributeSetter::SimulationAttributeSetter(QObject * parent)
	: QObject(parent)
{
	
}

SimulationAttributeSetter::~SimulationAttributeSetter()
{
	if (_registered) {
		_threadController->unregisterObserver(this);
	}
}

void SimulationAttributeSetter::init(SimulationContext * context)
{
	_threadController = static_cast<SimulationContextImpl*>(context)->getUnitThreadController();
	_grid = static_cast<SimulationContextImpl*>(context)->getUnitGrid();
	_threadController->registerObserver(this);
	_registered = true;
}

void SimulationAttributeSetter::setSimulationParameters(SimulationParameters const* parameters)
{
	_parameters = parameters;
	_updateSimulationParameters = true;
	if (_threadController->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAttributeSetter::unregister()
{
	_registered = false;
}

void SimulationAttributeSetter::accessToUnits()
{
	if (!_updateSimulationParameters) {
		return;
	}
	_updateSimulationParameters = false;

	IntVector2D gridSize = _grid->getSize();
	for (int gridX = 0; gridX < gridSize.x; ++gridX) {
		for (int gridY = 0; gridY < gridSize.y; ++gridY) {
			auto unitContext = _grid->getUnitOfGridPos({ gridX, gridY })->getContext();
			unitContext->setSimulationParameters(_parameters->clone());
		}
	}


}
