#include <QDir>
#include <QFile>
#include <gtest/gtest.h>

#include "ModelCpu/Cluster.h"
#include "ModelBasic/SimulationController.h"
#include "ModelCpu/UnitContext.h"
#include "ModelBasic/Settings.h"
#include "tests/TestSettings.h"


class ReplicatorTests : public ::testing::Test
{
public:
	ReplicatorTests();
	~ReplicatorTests();

protected:
	SimulationController* _simulationController = nullptr;
};


ReplicatorTests::ReplicatorTests()
{
/*
	_simulationController = new SimulationController();
*/
}

ReplicatorTests::~ReplicatorTests()
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
