#include <QDir>
#include <QFile>
#include <gtest/gtest.h>

#include "model/entities/CellCluster.h"
#include "model/SimulationController.h"
#include "model/context/UnitContext.h"
#include "model/ModelSettings.h"
#include "tests/TestSettings.h"


class SimulationReplicatorTest : public ::testing::Test
{
public:
	SimulationReplicatorTest();
	~SimulationReplicatorTest();

protected:
	SimulationController* _simulationController;
};


SimulationReplicatorTest::SimulationReplicatorTest()
{
	_simulationController = new SimulationController();
}

SimulationReplicatorTest::~SimulationReplicatorTest()
{
	delete _simulationController;
}

/*
TEST_F (IntegrationTestReplicator, testRunSimulation)
{
    QFile file(INTEGRATIONTEST_REPLICATOR_INIT);
	ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    QDataStream in(&file);
    _simulationController->loadUniverse(in);
    file.close();

    UnitContext* context = _simulationController->getSimulationContext();
    ASSERT_TRUE(!context->getClustersRef().empty());
    int replicatorSize = context->getClustersRef().at(0)->getCellsRef().size();
    for (int time = 0; time < INTEGRATIONTEST_REPLICATOR_TIMESTEPS; ++time) {
        _simulationController->requestNextTimestep();
    }

    int replicators = 0;
    foreach (CellCluster* cluster, context->getClustersRef()) {
        if (cluster->getCellsRef().size() >= replicatorSize)
            ++replicators;
    }
    ASSERT_TRUE(replicators >= 5);
}

*/
