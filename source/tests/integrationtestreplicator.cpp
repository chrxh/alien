#include "integrationtestreplicator.h"

#include "testsettings.h"
#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/simulationcontroller.h"
#include "model/metadatamanager.h"
#include "model/simulationsettings.h"

#include <QtTest/QtTest>

void IntegrationTestReplicator::initTestCase ()
{
    _simulationController = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD, this);
}

void IntegrationTestReplicator::testRunSimulation()
{
    QFile file(INTEGRATIONTEST_REPLICATOR_INIT);
    bool fileOpened = file.open(QIODevice::ReadOnly);
    if (fileOpened) {
        QDataStream in(&file);
        _simulationController->buildUniverse(in);
        file.close();
    }

    Grid* grid = _simulationController->getGrid();
    QVERIFY(!grid->getClusters().empty());
    int replicatorSize = grid->getClusters().at(0)->getCellsRef().size();
    for (int time = 0; time < INTEGRATIONTEST_REPLICATOR_TIMESTEPS; ++time) {
        _simulationController->requestNextTimestep();
    }

    int replicators = 0;
    foreach (CellCluster* cluster, grid->getClusters()) {
        if (cluster->getCellsRef().size() >= replicatorSize)
            ++replicators;
    }
    QVERIFY2(replicators > 8, "Not enough replicators.");
}

void IntegrationTestReplicator::cleanupTestCase()
{
}
