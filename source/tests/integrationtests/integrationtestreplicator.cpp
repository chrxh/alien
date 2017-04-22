#include <QDir>
#include <QFile>
#include <gtest/gtest.h>

#include "model/entities/cellcluster.h"
#include "model/simulationcontroller.h"
#include "model/simulationcontext.h"
#include "model/modelsettings.h"
#include "tests/settings.h"


class IntegrationTestReplicator : public ::testing::Test
{
public:
	IntegrationTestReplicator();
	~IntegrationTestReplicator();

protected:
	SimulationController* _simulationController;
};


IntegrationTestReplicator::IntegrationTestReplicator()
{
	_simulationController = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD);
}

IntegrationTestReplicator::~IntegrationTestReplicator()
{
	delete _simulationController;
}

TEST_F (IntegrationTestReplicator, testRunSimulation)
{
    QFile file(INTEGRATIONTEST_REPLICATOR_INIT);
	ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    QDataStream in(&file);
    _simulationController->loadUniverse(in);
    file.close();

    SimulationContext* context = _simulationController->getSimulationContext();
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

