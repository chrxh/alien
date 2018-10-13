#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelCpu/SimulationContextCpuImpl.h"
#include "ModelCpu/UnitGrid.h"
#include "ModelCpu/Unit.h"
#include "ModelCpu/UnitContext.h"
#include "ModelCpu/MapCompartment.h"
#include "ModelCpu/UnitThreadControllerImpl.h"
#include "ModelCpu/UnitThread.h"
#include "ModelCpu/SimulationControllerCpu.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"
#include "ModelCpu/ModelCpuData.h"
#include "ModelCpu/SimulationAccessCpu.h"

#include "tests/Predicates.h"

#include "IntegrationTestFramework.h"

class Benchmark
	: public IntegrationTestFramework
{
public:
	Benchmark();

protected:
	void createTestData(SimulationAccess* access);
};

Benchmark::Benchmark()
	: IntegrationTestFramework({ 12 * 33 * 3, 12 * 17 * 3 })
{
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

TEST_F(Benchmark, benchmarkOneThreadWithOneUnit)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, { 1,1 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(Benchmark, benchmarkOneThreadWithManyUnits)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(Benchmark, benchmarkFourThread)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(4, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(Benchmark, benchmarkEightThread)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(8, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);

	createTestData(access);
	runSimulation(20, controller);

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}
