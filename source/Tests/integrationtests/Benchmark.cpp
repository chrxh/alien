#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/Context/UnitGrid.h"
#include "Model/Context/Unit.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/MapCompartment.h"
#include "Model/Context/_Impl/UnitThreadControllerImpl.h"
#include "Model/Context/_Impl/UnitThread.h"
#include "Model/AccessPorts/SimulationAccess.h"

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
	IntVector2D _universeSize{ 12 * 33 * 3, 12 * 17 * 3 };
};

Benchmark::Benchmark()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
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
	DataChangeDescription desc;
	for (int i = 0; i < 20000 * 9; ++i) {
		desc.addNewParticle(ParticleChangeDescription().setPos(QVector2D(_numberGen->getRandomInt(_universeSize.x), _numberGen->getRandomInt(_universeSize.y)))
			.setVel(QVector2D(_numberGen->getRandomReal()*2.0 - 1.0, _numberGen->getRandomReal()*2.0 - 1.0))
			.setEnergy(50));
	}
	access->updateData(desc);
}

void Benchmark::runSimulation(int timesteps, SimulationController* controller)
{
	QEventLoop pause;
	int t = 0;
	controller->connect(controller, &SimulationController::nextTimestepCalculated, [&]() {
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
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto controller = facade->buildSimulationController(1, { 1, 1 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;
}

TEST_F(Benchmark, benchmarkOneThreadWithManyUnits)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto controller = facade->buildSimulationController(1, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;
}

TEST_F(Benchmark, benchmarkFourThread)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto controller = facade->buildSimulationController(4, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;
}

TEST_F(Benchmark, benchmarkEightThread)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto controller = facade->buildSimulationController(8, { 12, 6 }, _universeSize, _symbols, _parameters);
	auto access = facade->buildSimulationAccess(controller->getContext());

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;
}
