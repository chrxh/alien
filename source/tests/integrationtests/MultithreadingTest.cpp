#include <gtest/gtest.h>

#include "global/ServiceLocator.h"
#include "model/BuilderFacade.h"
#include "model/ModelSettings.h"
#include "model/SimulationController.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"

#include "tests/Predicates.h"

class MultithreadingTest : public ::testing::Test
{
public:
	MultithreadingTest();
	~MultithreadingTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	UnitGrid* _grid = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	IntVector2D _universeSize{ 1200, 600 };
	IntVector2D _compartmentSize;
};

MultithreadingTest::MultithreadingTest()
{
	_controller = new SimulationController();
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto metric = facade->buildSpaceMetric(_universeSize);
	auto symbols = ModelSettings::loadDefaultSymbolTable();
	auto parameters = ModelSettings::loadDefaultSimulationParameters();
	_context = facade->buildSimulationContext(4, _gridSize, metric, symbols, parameters, _controller);
	_controller->newUniverse(_context);
	_grid = _context->getUnitGrid();
	_compartmentSize = { _universeSize.x / _gridSize.x, _universeSize.y / _gridSize.y };
}

MultithreadingTest::~MultithreadingTest()
{
	delete _controller;
}
