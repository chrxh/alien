#ifndef ALIENTHREAD_H
#define ALIENTHREAD_H

#include <QThread>

class AlienCellCluster;
class AlienEnergy;
class AlienGrid;
class AlienThread : public QThread
{
    Q_OBJECT
public:
    AlienThread (QObject* parent = 0);
    ~AlienThread ();

    void init (AlienGrid* grid);
    QList< AlienCellCluster* >& getClusters ();
    QList< AlienEnergy* >& getEnergyParticles ();
//    void updateCluster (AlienCellCluster* cluster);

    qreal calcTransEnergy ();
    qreal calcRotEnergy ();
    qreal calcInternalEnergy ();

public slots:
    void setRandomSeed (uint seed);
    void calcNextTimestep ();

signals:
    void nextTimestepCalculated ();

protected:
    void debugCluster (AlienCellCluster* c, int s);

    AlienGrid* _grid;
    int _time;
};

#endif // ALIENTHREAD_H
