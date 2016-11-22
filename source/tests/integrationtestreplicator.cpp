#include "integrationtestreplicator.h"

#include "testsettings.h"
#include "model/simulationcontroller.h"
#include "model/modelfacade.h"
#include "model/entities/cellcluster.h"
#include "global/servicelocator.h"

#include <QtTest/QtTest>

void IntegrationTestReplicator::initTestCase ()
{
    _simulator = new SimulationController(QVector2D(500, 500), SimulationController::Threading::NO_EXTRA_THREAD);
}

void IntegrationTestReplicator::testLoadReplicator()
{

}

void IntegrationTestReplicator::testRunSimulation()
{
    //_simulator->requestNextTimestep();
}

void IntegrationTestReplicator::cleanupTestCase()
{
    delete _simulator;
}
