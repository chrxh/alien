#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelCpu/SimulationContextCpuImpl.h"
#include "ModelCpu/UnitGrid.h"
#include "ModelCpu/Unit.h"
#include "ModelCpu/UnitContext.h"
#include "ModelCpu/MapCompartment.h"

#include "tests/Predicates.h"

class MapCompartmentTest : public ::testing::Test
{
public:
	MapCompartmentTest();
	~MapCompartmentTest();

protected:
	IntVector2D correctUniversePosition(IntVector2D const& pos);

	SimulationController* _controller = nullptr;
	SimulationContextCpuImpl* _context = nullptr;
	UnitGrid* _grid = nullptr;
	const IntVector2D _gridSize{ 6, 8 };
	const IntVector2D _universeSize{ 1200, 800 };
	IntVector2D _compartmentSize;
};

MapCompartmentTest::MapCompartmentTest()
{
	ModelBasicBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	auto symbols = facade->buildDefaultSymbolTable();
	auto parameters = facade->buildDefaultSimulationParameters();

	_controller = facade->buildSimulationController(4, _gridSize, _universeSize, symbols, parameters);
	_context = static_cast<SimulationContextCpuImpl*>(_controller->getContext());

	_grid = _context->getUnitGrid();
	_compartmentSize = { _universeSize.x / _gridSize.x, _universeSize.y / _gridSize.y };
}

MapCompartmentTest::~MapCompartmentTest()
{
	delete _controller;
}

IntVector2D MapCompartmentTest::correctUniversePosition(IntVector2D const & pos)
{
	return{ ((pos.x % _universeSize.x) + _universeSize.x) % _universeSize.x, ((pos.y % _universeSize.y) + _universeSize.y) % _universeSize.y };
}

TEST_F(MapCompartmentTest, testCompartmentRect)
{
	for (int x = 0; x < _gridSize.x; ++x) {
		for (int y = 0; y < _gridSize.y; ++y) {
			auto unit = _grid->getUnitOfGridPos({ x, y });
			auto unitContext = unit->getContext();
			auto compartment = unitContext->getMapCompartment();
			ASSERT_PRED2(predEqualIntVector, _compartmentSize, compartment->getSize()) << "Compartment size does not match.";

			IntVector2D centerPoint = { x * _compartmentSize.x + _compartmentSize.x/2, y*_compartmentSize.y + _compartmentSize.y/2 };
			IntVector2D leftPointInside = correctUniversePosition({ centerPoint.x - _compartmentSize.x / 2, centerPoint.y });
			IntVector2D rightPointInside = correctUniversePosition({ centerPoint.x + _compartmentSize.x / 2 - 1, centerPoint.y });
			IntVector2D upperPointInside = correctUniversePosition({ centerPoint.x, centerPoint.y - _compartmentSize.y / 2 });
			IntVector2D lowerPointInside = correctUniversePosition({ centerPoint.x, centerPoint.y + _compartmentSize.y / 2 - 1 });
			auto const& msgInteriorPoint = "An actual interior point is not contained in unit compartment.";
			ASSERT_TRUE(compartment->isPointInCompartment(centerPoint)) << msgInteriorPoint;
			ASSERT_TRUE(compartment->isPointInCompartment(leftPointInside)) << msgInteriorPoint;
			ASSERT_TRUE(compartment->isPointInCompartment(rightPointInside)) << msgInteriorPoint;
			ASSERT_TRUE(compartment->isPointInCompartment(upperPointInside)) << msgInteriorPoint;
			ASSERT_TRUE(compartment->isPointInCompartment(lowerPointInside)) << msgInteriorPoint;

			IntVector2D leftPointOutside = correctUniversePosition({ centerPoint.x - _compartmentSize.x / 2 - 1, centerPoint.y });
			IntVector2D rightPointOutside = correctUniversePosition({ centerPoint.x + _compartmentSize.x / 2, centerPoint.y });
			IntVector2D upperPointOutside = correctUniversePosition({ centerPoint.x, centerPoint.y - _compartmentSize.y / 2  - 1});
			IntVector2D lowerPointOutside = correctUniversePosition({ centerPoint.x, centerPoint.y + _compartmentSize.y / 2 });
			auto const& msgExteriorPoint = "An actual exterior point is contained in unit compartment.";
			ASSERT_FALSE(compartment->isPointInCompartment(leftPointOutside)) << msgExteriorPoint;
			ASSERT_FALSE(compartment->isPointInCompartment(rightPointOutside)) << msgExteriorPoint;
			ASSERT_FALSE(compartment->isPointInCompartment(upperPointOutside)) << msgExteriorPoint;
			ASSERT_FALSE(compartment->isPointInCompartment(lowerPointOutside)) << msgExteriorPoint;
		}
	}
}

TEST_F(MapCompartmentTest, testCoordinateConversion)
{
	for (int x = 0; x < _gridSize.x; ++x) {
		for (int y = 0; y < _gridSize.y; ++y) {
			auto unit = _grid->getUnitOfGridPos({ x, y });
			auto unitContext = unit->getContext();
			auto compartment = unitContext->getMapCompartment();

			IntVector2D centerPointAbs = correctUniversePosition({ x * _compartmentSize.x + _compartmentSize.x / 2, y * _compartmentSize.y + _compartmentSize.y / 2 });
			IntVector2D leftPointAbs = correctUniversePosition({ centerPointAbs.x - _compartmentSize.x / 2, centerPointAbs.y });
			IntVector2D rightPointAbs = correctUniversePosition({ centerPointAbs.x + _compartmentSize.x / 2 - 1, centerPointAbs.y });
			IntVector2D upperPointAbs = correctUniversePosition({ centerPointAbs.x, centerPointAbs.y - _compartmentSize.y / 2 });
			IntVector2D lowerPointAbs = correctUniversePosition({ centerPointAbs.x, centerPointAbs.y + _compartmentSize.y / 2 - 1});

			IntVector2D expectedCenterPointRel = { _compartmentSize.x / 2, _compartmentSize.y / 2 };
			IntVector2D expectedLeftPointRel = { 0, _compartmentSize.y / 2 };
			IntVector2D expectedRightPointRel = { _compartmentSize.x - 1, _compartmentSize.y / 2 };
			IntVector2D expectedUpperPointRel = { _compartmentSize.x / 2, 0 };
			IntVector2D expectedLowerPointRel = { _compartmentSize.x / 2, _compartmentSize.y - 1 };

			compartment->convertAbsToRelPosition(centerPointAbs);
			compartment->convertAbsToRelPosition(leftPointAbs);
			compartment->convertAbsToRelPosition(rightPointAbs);
			compartment->convertAbsToRelPosition(upperPointAbs);
			compartment->convertAbsToRelPosition(lowerPointAbs);

			ASSERT_PRED2(predEqualIntVector, expectedCenterPointRel, centerPointAbs);
			ASSERT_PRED2(predEqualIntVector, expectedLeftPointRel, leftPointAbs);
			ASSERT_PRED2(predEqualIntVector, expectedRightPointRel, rightPointAbs);
			ASSERT_PRED2(predEqualIntVector, expectedUpperPointRel, upperPointAbs);
			ASSERT_PRED2(predEqualIntVector, expectedLowerPointRel, lowerPointAbs);
		}
	}
}

TEST_F(MapCompartmentTest, testNeighborContext)
{
	vector<vector<UnitContext*>> contexts(_gridSize.x, vector<UnitContext*>(_gridSize.y));
	for (int x = 0; x < _gridSize.x; ++x) {
		for (int y = 0; y < _gridSize.y; ++y) {
			contexts[x][y] = _grid->getUnitOfGridPos({ x, y })->getContext();
		}
	}
	for (int x = 0; x < _gridSize.x; ++x) {
		for (int y = 0; y < _gridSize.y; ++y) {
			auto context = contexts[x][y];
			auto compartment = context->getMapCompartment();
			IntVector2D gridPosLeft = { (x - 1 + _gridSize.x) % _gridSize.x, y };
			IntVector2D gridPosRight = { (x + 1) % _gridSize.x , y };
			IntVector2D gridPosUpper = { x, (y - 1 + _gridSize.y) % _gridSize.y};
			IntVector2D gridPosLower = { x, (y + 1) % _gridSize.y };
			IntVector2D pointPosLeft = { gridPosLeft.x * _compartmentSize.x + _compartmentSize.x / 2, gridPosLeft.y * _compartmentSize.y + _compartmentSize.y / 2 };
			IntVector2D pointPosRight = { gridPosRight.x * _compartmentSize.x + _compartmentSize.x / 2, gridPosRight.y * _compartmentSize.y + _compartmentSize.y / 2 };
			IntVector2D pointPosUpper = { gridPosUpper.x * _compartmentSize.x + _compartmentSize.x / 2, gridPosUpper.y * _compartmentSize.y + _compartmentSize.y / 2 };
			IntVector2D pointPosLower = { gridPosLower.x * _compartmentSize.x + _compartmentSize.x / 2, gridPosLower.y * _compartmentSize.y + _compartmentSize.y / 2 };
			auto contextLeft = contexts[gridPosLeft.x][gridPosLeft.y];
			auto contextRight = contexts[gridPosRight.x][gridPosRight.y];
			auto contextUpper = contexts[gridPosUpper.x][gridPosUpper.y];
			auto contextLower = contexts[gridPosLower.x][gridPosLower.y];
			ASSERT_EQ(contextLeft, compartment->getNeighborContext(pointPosLeft)) << "Left UnitContext does not match at gridPos = (" << x << ", " << y << ").";
			ASSERT_EQ(contextRight, compartment->getNeighborContext(pointPosRight)) << "Right UnitContext does not match at gridPos = (" << x << ", " << y << ").";
			ASSERT_EQ(contextUpper, compartment->getNeighborContext(pointPosUpper)) << "Upper UnitContext does not match at gridPos = (" << x << ", " << y << ").";
			ASSERT_EQ(contextLower, compartment->getNeighborContext(pointPosLower)) << "Lower UnitContext does not match at gridPos = (" << x << ", " << y << ").";
		}
	}
}
