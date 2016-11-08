#ifndef ALIENSIMULATOR_H
#define ALIENSIMULATOR_H

#include "entities/aliencellto.h"

#include <QObject>
#include <QVector3D>

class QTimer;
class AlienCell;
class AlienCellCluster;
class AlienEnergy;
class AlienGrid;
class AlienThread;
class Visualizer;

class AlienSimulator  : public QObject
{
    Q_OBJECT
public:
    AlienSimulator (int sizeX, int sizeY, QObject* parent = 0);
    ~AlienSimulator ();

    QMap< QString, qreal > getMonitorData ();

    //universe manipulation tools
    void newUniverse (qint32 sizeX, qint32 sizeY);
    void serializeUniverse (QDataStream& stream);
    void buildUniverse (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap, QMap< quint64, quint64 >& oldNewCellIdMap);
    qint32 getUniverseSizeX();
    qint32 getUniverseSizeY ();
    void addBlockStructure (QVector3D center, int numCellX, int numCellY, QVector3D dist, qreal energy);
    void addHexagonStructure (QVector3D center, int numLayers, qreal dist, qreal energy);
    void addRandomEnergy (qreal energy, qreal maxEnergyPerParticle);

    //selection manipulation Tools
    void serializeCell (QDataStream& stream, AlienCell* cell, quint64& clusterId, quint64& cellId);
    void serializeExtendedSelection (QDataStream& stream,
                                    const QList< AlienCellCluster* >& clusters,
                                    const QList< AlienEnergy* >& es,
                                    QList< quint64 >& clusterIds,
                                    QList< quint64 >& cellIds);
    void buildCell (QDataStream& stream,                //returns a map which maps to old to the new cell and cluster ids
                    QVector3D pos,
                    AlienCellCluster*& newCluster,
                    QMap< quint64, quint64 >& oldNewClusterIdMap,
                    QMap< quint64, quint64 >& oldNewCellIdMap,
                    bool drawToMap = true);
    void buildExtendedSelection (QDataStream& stream,   //returns a map which maps to old to the new cell and cluster ids
                                QVector3D pos,
                                QList< AlienCellCluster* >& newClusters,
                                QList< AlienEnergy* >& newEnergyParticles,
                                QMap< quint64, quint64 >& oldNewClusterIdMap,
                                QMap< quint64, quint64 >& oldNewCellIdMap,
                                bool drawToMap = true);
public slots:
    void delSelection (QList< AlienCell* > cells,
                      QList< AlienEnergy* > es);
    void delExtendedSelection (QList< AlienCellCluster* > clusters,
                         QList< AlienEnergy* > es);
public:
    void rotateExtendedSelection (qreal angle, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es);
    void setVelocityXExtendedSelection (qreal velX, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es);
    void setVelocityYExtendedSelection (qreal velY, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es);
    void setAngularVelocityExtendedSelection (qreal angVel, const QList< AlienCellCluster* >& clusters);
    QVector3D getCenterPosExtendedSelection (const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es);
    void drawToMapExtendedSelection (const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es);

    //cell/particle manipulation tools
public slots:
    void newCell (QVector3D pos);
    void newEnergyParticle (QVector3D pos);
    void updateCell (QList< AlienCell* > cells,
                     QList< AlienCellTO > newCellsData,
                     bool clusterDataChanged);

    //misc
public slots:
    void setRun (bool run);
    void forceFps (int fps);
    void requestNextTimestep ();

    void updateUniverse ();

signals:
    void setRandomSeed (uint seed);
    void calcNextTimestep ();
    void cellCreated (AlienCell* cell);
    void energyParticleCreated (AlienEnergy* cell);
    void reclustered (QList< AlienCellCluster* > clusters);
    void universeUpdated (AlienGrid* grid, bool force);
    void computerCompilationReturn (bool error, int line);

protected slots:
    virtual void forceFpsTimerSlot ();
    virtual void nextTimestepCalculated ();

protected:
    QTimer* _forceFpsTimer;
    bool _run;
    int _fps;
    bool _calculating;
    //bool _editMode;
    quint64 _frame;
    int _newCellTokenAccessNumber;

    AlienGrid* _grid;
    AlienThread* _thread;
};

#endif // ALIENSIMULATOR_H
