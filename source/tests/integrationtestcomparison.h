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
    void testRunSimulations ();
    void cleanupTestCase();

private:
    SimulationController* _simController;
};

#endif // INTEGRATIONTESTCOMPARISON_H
