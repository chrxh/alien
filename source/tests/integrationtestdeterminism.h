#ifndef INTEGRATIONTESTDETERMINISM_H
#define INTEGRATIONTESTDETERMINISM_H

#include <QObject>
#include <QVector3D>
#include <QMutex>

class SimulationController;
class Grid;

class IntegrationTestDeterminism : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase();
    void testRunSimulations ();
    void cleanupTestCase();

private:
    SimulationController* _simController;
};

#endif // INTEGRATIONTESTDETERMINISM_H
