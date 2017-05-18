#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "model/BuilderFacade.h"
#include "model/Settings.h"
#include "model/SimulationController.h"
#include "model/Context/SimulationContext.h"
#include "model/Context/SimulationParameters.h"
#include "model/Context/UnitGrid.h"
#include "model/Context/Unit.h"
#include "model/Context/UnitContext.h"
#include "model/Context/MapCompartment.h"
#include "model/Context/_Impl/UnitThreadControllerImpl.h"
#include "model/Context/_Impl/UnitThread.h"
#include "model/AccessPorts/SimulationAccess.h"

#include "tests/Predicates.h"

class Benchmark : public ::testing::Test
{
public:
	Benchmark();
	~Benchmark();

protected:
	void createTestData(SimulationAccess* access);
	void runSimulation(int timesteps, SimulationController* controller);

	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	SymbolTable* _symbols = nullptr;
	IntVector2D _universeSize{ 1200, 600 };
};

Benchmark::Benchmark()
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_symbols = facade->buildDefaultSymbolTable();
	_parameters = facade->buildDefaultSimulationParameters();
	_numberGen = factory->buildRandomNumberGenerator();
	_numberGen->init(123123, 0);
}

Benchmark::~Benchmark()
{
	delete _numberGen;
}

void Benchmark::createTestData(SimulationAccess * access)
{
	DataDescription desc;
	for (int i = 0; i < 40000; ++i) {
		desc.addEnergyParticle(EnergyParticleDescription().setPos(QVector2D(_numberGen->getRandomInt(_universeSize.x), _numberGen->getRandomInt(_universeSize.y)))
			.setVel(QVector2D(_numberGen->getRandomReal()*2.0 - 1.0, _numberGen->getRandomReal()*2.0 - 1.0))
			.setEnergy(50));
	}
	access->updateData(desc);
}

void Benchmark::runSimulation(int timesteps, SimulationController* controller)
{
	QEventLoop pause;
	int t = 0;
	controller->connect(controller, &SimulationController::timestepCalculated, [&]() {
		if (++t == timesteps) {
			controller->setRun(false);
			pause.quit();
		}
	});
	controller->setRun(true);
	pause.exec();
}

TEST_F(Benchmark, benchmarkOneThreadWithOneUnit)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto controller = facade->buildSimulationController(1, { 1, 1 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(300, controller);

	delete controller;
}

TEST_F(Benchmark, benchmarkOneThreadWithManyUnits)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto controller = facade->buildSimulationController(1, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(300, controller);

	delete controller;
}

TEST_F(Benchmark, benchmarkFourThread)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto controller = facade->buildSimulationController(4, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(300, controller);

	delete controller;
}

TEST_F(Benchmark, benchmarkEightThread)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	auto controller = facade->buildSimulationController(8, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(300, controller);

	delete controller;
}
