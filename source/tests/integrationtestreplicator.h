#ifndef INTEGRATIONTESTREPLICATOR_H
#define INTEGRATIONTESTREPLICATOR_H

#include <QObject>

class SimulationController;

class IntegrationTestReplicator : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase();
    void testLoadReplicator();
    void testRunSimulation();
    void cleanupTestCase();

private:
    SimulationController* _simulator;
};


#endif // INTEGRATIONTESTREPLICATOR_H
