#include "tests/settings.h"
#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/simulationcontroller.h"
#include "model/metadatamanager.h"
#include "model/simulationsettings.h"

#include <gtest/gtest.h>
#include <QFile>
#include <QDir>

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
    _simulationController->buildUniverse(in);
    file.close();

    Grid* grid = _simulationController->getGrid();
    ASSERT_TRUE(!grid->getClusters().empty());
    int replicatorSize = grid->getClusters().at(0)->getCellsRef().size();
    for (int time = 0; time < INTEGRATIONTEST_REPLICATOR_TIMESTEPS; ++time) {
        _simulationController->requestNextTimestep();
    }

    int replicators = 0;
    foreach (CellCluster* cluster, grid->getClusters()) {
        if (cluster->getCellsRef().size() >= replicatorSize)
            ++replicators;
    }
    ASSERT_TRUE(replicators > 8);
}

