#include <gtest/gtest.h>

#include <QEventLoop>

#include "global/ServiceLocator.h"
#include "model/BuilderFacade.h"
#include "model/ModelSettings.h"
#include "model/SimulationController.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"
#include "model/context/_impl/UnitThreadControllerImpl.h"
#include "model/context/_impl/UnitThread.h"

#include "tests/Predicates.h"

class MultithreadingTest : public ::testing::Test
{
public:
	MultithreadingTest();
	~MultithreadingTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	UnitThreadControllerImpl* _threadController = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	IntVector2D _universeSize{ 1200, 600 };
	IntVector2D _compartmentSize;
};

MultithreadingTest::MultithreadingTest()
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto metric = facade->buildSpaceMetric(_universeSize);
	auto symbols = facade->buildDefaultSymbolTable();
	auto parameters = facade->buildDefaultSimulationParameters();
	_context = static_cast<SimulationContext*>(facade->buildSimulationContext(4, _gridSize, metric, symbols, parameters));
	_controller = facade->buildSimulationController(_context);
	_threadController = static_cast<UnitThreadControllerImpl*>(_context->getUnitThreadController());
}

MultithreadingTest::~MultithreadingTest()
{
	delete _controller;
}

TEST_F(MultithreadingTest, testThreads)
{
	QEventLoop pause;
	_threadController->connect(_threadController, &UnitThreadController::timestepCalculated, &pause, &QEventLoop::quit);
	_threadController->start();
	pause.exec();

	for (auto const& threadAndCalcSignal : _threadController->_threadsAndCalcSignals) {
		ASSERT_TRUE(threadAndCalcSignal.thr->isFinished()) << "One thread is not finished.";
	}
}