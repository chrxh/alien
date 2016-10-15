#ifndef ALIENSPACE_H
#define ALIENSPACE_H

#include "aliencell.h"
#include "../../globaldata/simulationsettings.h"

#include <QObject>
#include <QVector3D>
#include <QMutex>
#include <QMap>
#include <QSet>
#include <QtCore/qmath.h>

//class AlienCell;
class AlienEnergy;
class AlienCellCluster;
class AlienGrid : public QObject
{
    Q_OBJECT
public:
    AlienGrid (QObject* parent = 0);
    ~AlienGrid ();

    void init (qint32 sizeX, qint32 sizeY);
    void reinit (qint32 sizeX, qint32 sizeY);
    void lockData ();
    void unlockData ();

    //access functions to all entities
    QList< AlienCellCluster* >& getClusters ();
    QList< AlienEnergy* >& getEnergyParticles ();
    QSet< quint64 > getAllCellIds () const;
    void clearGrids ();
    qint32 getSizeX() const;
    qint32 getSizeY() const;

    //cell grid access functions
    void setCell (QVector3D pos, AlienCell* cell);
    void removeCell (QVector3D pos);
    void removeCellIfPresent (QVector3D pos, AlienCell* cell);
    AlienCell* getCell (QVector3D pos) const;
    AlienCell* getCellFast (const int& x, const int& y) const;

    //location functions
    QSet< AlienCellCluster* > getNearbyClusters (const QVector3D& pos, qreal r) const;
    AlienCellCluster* getNearbyClusterFast (const QVector3D& pos, qreal r, qreal minMass, qreal maxMass, AlienCellCluster* exclude) const;
    using CellSelectFunction = bool(*)(AlienCell*);
    QList< AlienCell* > getNearbySpecificCells (const QVector3D& pos, qreal r, CellSelectFunction selection) const;

    //energy grid access functions
    void removeEnergy (QVector3D pos, AlienEnergy* energy);
    AlienEnergy* getEnergyFast (const int& x, const int& y) const;
    void setEnergy(QVector3D pos, AlienEnergy* energy);
    AlienEnergy* getEnergy (QVector3D pos) const;

    //auxiliary functions
    void correctPosition (QVector3D& pos) const;
    void correctDisplacement (QVector3D& displacement) const;
    QVector3D displacement (QVector3D fromPoint, QVector3D toPoint) const;
    QVector3D displacement (AlienCell* fromCell, AlienCell* toCell) const;
    qreal distance (AlienCell* fromCell, AlienCell* toCell) const;

    //(de)serialisation functions
    void serializeSize (QDataStream& stream) const;
    void serializeMap (QDataStream& stream) const;
    void buildEmptyMap (QDataStream& stream);
    void buildMap (QDataStream& stream, const QMap< quint64, AlienCell* >& oldIdCellMap, const QMap< quint64, AlienEnergy* >& oldIdEnergyMap);

private:
    QMutex _mutex;
    qint32 _sizeX;
    qint32 _sizeY;
    AlienCell*** _cellGrid;
    AlienEnergy*** _energyGrid;

    QList< AlienCellCluster* > _clusters;
    QList< AlienEnergy* > _energyParticles;
};

/******************
 * inline functions
 ******************/
inline QList< AlienCellCluster* >& AlienGrid::getClusters ()
{
    return _clusters;
}

inline QList< AlienEnergy* >& AlienGrid::getEnergyParticles ()
{
    return _energyParticles;
}

inline void AlienGrid::setCell (QVector3D pos, AlienCell* cell)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    _cellGrid[x][y] = cell;
}

inline AlienCell* AlienGrid::getCell (QVector3D pos) const
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    return _cellGrid[x][y];
}

inline AlienCell* AlienGrid::getCellFast (const int &x, const int& y) const
{
    return _cellGrid[x][y];
}

inline AlienEnergy* AlienGrid::getEnergyFast (const int& x, const int& y) const
{
    return _energyGrid[x][y];
}

inline QSet< AlienCellCluster* > AlienGrid::getNearbyClusters (const QVector3D& pos, qreal r) const
{
    QSet< AlienCellCluster* > clusters;
//    int r = qFloor(simulationParameters.CRIT_CELL_DIST_MAX+1.0);
    int rc = qCeil(r);
    for(int rx = pos.x()-rc; rx < pos.x()+rc+1; ++rx)
        for(int ry = pos.y()-rc; ry < pos.y()+rc+1; ++ry) {
            if( QVector3D(static_cast<float>(rx)-pos.x(),static_cast<float>(ry)-pos.y(),0).length() < r+ALIEN_PRECISION ) {
                AlienCell* cell(getCell(QVector3D(rx,ry,0)));
                if( cell )
                    clusters << cell->getCluster();
            }
        }
    return clusters;
}

inline QList< AlienCell* > AlienGrid::getNearbySpecificCells (const QVector3D& pos, qreal r, CellSelectFunction selection) const
{
    QList< AlienCell* > cells;
    int rCeil = qCeil(r);
    for(int scanX = pos.x()-rCeil; scanX < pos.x()+rCeil+1; ++scanX)
        for(int scanY = pos.y()-rCeil; scanY < pos.y()+rCeil+1; ++scanY) {
            if( QVector3D(static_cast<float>(scanX)-pos.x(),static_cast<float>(scanY)-pos.y(),0).length() < r+ALIEN_PRECISION ) {
                AlienCell* cell(getCell(QVector3D(scanX, scanY,0)));
                if( cell ) {
                    if( selection(cell) ) {
                        cells << cell;
                    }
                }
            }
        }
    return cells;
}

inline void AlienGrid::removeEnergy (QVector3D pos, AlienEnergy* energy)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    if( _energyGrid[x][y] == energy )
        _energyGrid[x][y] = 0;
}

inline void AlienGrid::setEnergy (QVector3D pos, AlienEnergy* energy)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    _energyGrid[x][y] = energy;
}

inline AlienEnergy* AlienGrid::getEnergy (QVector3D pos) const
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    return _energyGrid[x][y];
}

#endif // ALIENSPACE_H
