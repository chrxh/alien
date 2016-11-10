#ifndef THREAD_H
#define THREAD_H

#include <QThread>

class CellCluster;
class EnergyParticle;
class Grid;
class Thread : public QThread
{
    Q_OBJECT
public:
    Thread (QObject* parent = 0);
    ~Thread ();

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

#endif // THREAD_H
