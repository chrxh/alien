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
    bool compareClusterSizes ();
    QList<int> getAbnormalClusterNumbers (int timestep);

    SimulationController* _simController1;
    SimulationController* _simController2;
    quint64 _tag1 = 0;
    quint64 _tag2 = 0;
};

#endif // INTEGRATIONTESTDETERMINISM_H
