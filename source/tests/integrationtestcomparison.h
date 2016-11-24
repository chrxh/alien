#ifndef INTEGRATIONTESTCOMPARISON_H
#define INTEGRATIONTESTCOMPARISON_H

#include <QObject>
#include <QVector3D>
#include <QMutex>

class SimulationController;
class Grid;

class IntegrationTestComparison : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase();
    void testRunAndCompareSimulation ();
    void cleanupTestCase();

private:
    void test();

    SimulationController* _simulationController;
};

#endif // INTEGRATIONTESTCOMPARISON_H
