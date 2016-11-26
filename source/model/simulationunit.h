#ifndef SIMULATIONUNIT_H
#define SIMULATIONUNIT_H

#include "definitions.h"
#include <QThread>

class SimulationUnit : public QObject
{
    Q_OBJECT
public:
    SimulationUnit (QObject* parent = 0);
    ~SimulationUnit ();

    void init (Grid* grid);
    QList< CellCluster* >& getClusters ();
    QList< EnergyParticle* >& getEnergyParticles ();
//    void updateCluster (CellCluster* cluster);

    qreal calcTransEnergy ();
    qreal calcRotEnergy ();
    qreal calcInternalEnergy ();

public slots:
    void setRandomSeed (uint seed);
    void calcNextTimestep ();

signals:
    void nextTimestepCalculated ();

protected:
    void debugCluster (CellCluster* c, int s);

    Grid* _grid;
    int _time;
};

#endif // SIMULATIONUNIT_H
