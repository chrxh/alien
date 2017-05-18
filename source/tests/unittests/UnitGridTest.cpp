#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "model/BuilderFacade.h"
#include "model/Settings.h"
#include "model/SimulationController.h"
#include "model/Context/SimulationContext.h"
#include "model/Context/UnitGrid.h"
#include "model/Context/Unit.h"
#include "model/Context/UnitContext.h"
#include "model/Context/MapCompartment.h"

#include "tests/Predicates.h"

class UnitGridTest : public ::testing::Test
{
public:
	UnitGridTest();
	~UnitGridTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	UnitGrid* _grid = nullptr;
	const IntVector2D _gridSize{ 6, 6 };
	const IntVector2D _universeSize{ 1200, 600 };
	IntVector2D _compartmentSize;
};

UnitGridTest::UnitGridTest()
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto symbols = facade->buildDefaultSymbolTable();
	auto parameters = facade->buildDefaultSimulationParameters();

	_controller = facade->buildSimulationController(4, _gridSize, _universeSize, symbols, parameters);
	_context = static_cast<SimulationContext*>(_controller->getContext());

	_grid = _context->getUnitGrid();
	_compartmentSize = { _universeSize.x / _gridSize.x, _universeSize.y / _gridSize.y };
}

UnitGridTest::~UnitGridTest()
{
	delete _controller;
}

TEST_F(UnitGridTest, testGridSize)
{
	ASSERT_PRED2(predEqualIntVector, _gridSize, _grid->getSize());
}

TEST_F(UnitGridTest, testCompartmentRects)
{
	for (int x = 0; x < _gridSize.x; ++x) {
		for (int y = 0; y < _gridSize.y; ++y) {
			auto rect = _grid->calcCompartmentRect({ x, y });
			IntVector2D expectedRectP1 = { x*_compartmentSize.x, y*_compartmentSize.y };
			IntVector2D expectedRectP2 = { (x + 1)*_compartmentSize.x - 1, (y + 1)*_compartmentSize.y - 1 };
			ASSERT_PRED2(predEqualIntVector, expectedRectP1, rect.p1);
			ASSERT_PRED2(predEqualIntVector, expectedRectP2, rect.p2);
		}
	}
}
