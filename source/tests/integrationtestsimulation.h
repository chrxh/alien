#ifndef INTEGRATIONTESTSIMULATION_H
#define INTEGRATIONTESTSIMULATION_H

#include "testsettings.h"
#include "model/simulationcontroller.h"
#include "model/modelfacade.h"
#include "model/entities/cellcluster.h"
#include "global/servicelocator.h"

#include <QtTest/QtTest>

class IntegrationTestSimulation : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase()
    {
        _simulator = new SimulationController(500, 500, SimulationController::Threading::SINGLE);
    }

    void testLoadReplicator()
    {

    }

    void testRunSimulation()
    {
        _simulator->requestNextTimestep();
    }

    void cleanupTestCase()
    {
        delete _simulator;
    }

private:
    SimulationController* _simulator;
};


#endif // INTEGRATIONTESTSIMULATION_H
