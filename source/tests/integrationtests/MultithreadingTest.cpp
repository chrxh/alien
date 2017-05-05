#include <gtest/gtest.h>

#include <QEventLoop>

#include "global/ServiceLocator.h"
#include "global/NumberGenerator.h"
#include "model/BuilderFacade.h"
#include "model/ModelSettings.h"
#include "model/SimulationController.h"
#include "model/context/SimulationContext.h"
#include "model/context/SimulationParameters.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"
#include "model/context/_impl/UnitThreadControllerImpl.h"
#include "model/context/_impl/UnitThread.h"
#include "model/SimulationAccessApi.h"

#include "tests/Predicates.h"

class MultithreadingTest : public ::testing::Test
{
public:
	MultithreadingTest();
	~MultithreadingTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SimulationParameters* _parameters = nullptr;
	UnitThreadControllerImpl* _threadController = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	IntVector2D _universeSize{ 600, 300 };
	IntVector2D _compartmentSize;
};

MultithreadingTest::MultithreadingTest()
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto metric = facade->buildSpaceMetric(_universeSize);
	auto symbols = facade->buildDefaultSymbolTable();
	_parameters = facade->buildDefaultSimulationParameters();
	_context = static_cast<SimulationContext*>(facade->buildSimulationContext(4, _gridSize, metric, symbols, _parameters));
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
	_controller->connect(_controller, &SimulationController::timestepCalculated, &pause, &QEventLoop::quit);
	_controller->calculateSingleTimestep();
	pause.exec();

	for (auto const& threadAndCalcSignal : _threadController->_threadsAndCalcSignals) {
		ASSERT_TRUE(threadAndCalcSignal.thr->isFinished()) << "One thread is not finished.";
	}
}


TEST_F(MultithreadingTest, testOneCellMovement)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto access = facade->buildSimulationAccess(_context);

	_parameters->radiationProb = 0.0;

	CellDescription desc;
	desc.pos = QVector3D(100, 50, 0);
	desc.vel = QVector3D(1, 0.5, 0);
	desc.energy = _parameters->cellCreationEnergy;
	access->addCell(desc);

	QEventLoop pause;
	int timesteps = 0;
	_controller->connect(_controller, &SimulationController::timestepCalculated, [&]() {
		if (++timesteps == 300) {
			_controller->setRun(false);
			pause.quit();
		}
	});
	_controller->setRun(true);
	pause.exec();
}


TEST_F(MultithreadingTest, testManyCellsMovement)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto access = facade->buildSimulationAccess(_context);
	for (int i = 0; i < 10000; ++i) {
		CellDescription desc;
		desc.pos = QVector3D(NumberGenerator::getInstance().random(_universeSize.x), NumberGenerator::getInstance().random(_universeSize.y), 0);
		desc.vel = QVector3D(NumberGenerator::getInstance().random()-0.5, NumberGenerator::getInstance().random() - 0.5, 0);
		desc.energy = 100;
		access->addCell(desc);
	}

	QEventLoop pause;
	int timesteps = 0;
	_controller->connect(_controller, &SimulationController::timestepCalculated, [&]() {
		if (++timesteps == 50) {
			_controller->setRun(false);
			pause.quit();
		}
	});
	_controller->setRun(true);
	pause.exec();
}

