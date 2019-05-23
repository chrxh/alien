#include <gtest/gtest.h>

#include <QElapsedTimer>
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

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class CpuBenchmark
	: public IntegrationTestFramework
{
public:
    CpuBenchmark() : IntegrationTestFramework({ 2004, 1002 })
    { }

protected:
	void createTestData(SimulationAccess* access);
};


void CpuBenchmark::createTestData(SimulationAccess * access)
{
	DataDescription desc;

    for (int i = 0; i < 1000; ++i) {
        desc.addCluster(createRectangularCluster({ 7, 40 },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y)) },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(-1, 1)),
            static_cast<float>(_numberGen->getRandomReal(-1, 1)) }
        ));
    }
	access->updateData(desc);
}

TEST_F(CpuBenchmark, benchmarkOneThreadWithOneUnit)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, { 1,1 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);
	_numberGen = controller->getContext()->getNumberGenerator();

	createTestData(access);

    QElapsedTimer timer;
    timer.start();
	IntegrationTestHelper::runSimulation(200, controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(CpuBenchmark, benchmarkOneThreadWithManyUnits)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);
	_numberGen = controller->getContext()->getNumberGenerator();

	createTestData(access);
    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(CpuBenchmark, benchmarkFourThread)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(4, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);
	_numberGen = controller->getContext()->getNumberGenerator();

	createTestData(access);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}

TEST_F(CpuBenchmark, benchmarkEightThread)
{
	auto cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto controller = cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(8, { 12, 6 }));
	auto access = cpuFacade->buildSimulationAccess();
	access->init(controller);
	_numberGen = controller->getContext()->getNumberGenerator();

	createTestData(access);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;

	delete access;
	delete controller;

	EXPECT_TRUE(true);
}
