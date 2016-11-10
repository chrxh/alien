#ifndef GRID_H
#define GRID_H

#include "cell.h"
#include "model/simulationsettings.h"

#include <QObject>
#include <QVector3D>
#include <QMutex>
#include <QMap>
#include <QSet>
#include <QtCore/qmath.h>

//class Cell;
class EnergyParticle;
class CellCluster;
class Grid : public QObject
{
    Q_OBJECT
public:
    Grid (QObject* parent = 0);
    ~Grid ();

    void init (qint32 sizeX, qint32 sizeY);
    void reinit (qint32 sizeX, qint32 sizeY);
    void lockData ();
    void unlockData ();

    //access functions to all entities
    QList< CellCluster* >& getClusters ();
    QList< EnergyParticle* >& getEnergyParticles ();
    QSet< quint64 > getAllCellIds () const;
    void clearGrids ();
    qint32 getSizeX() const;
    qint32 getSizeY() const;

    //cell grid access functions
    void setCell (QVector3D pos, Cell* cell);
    void removeCell (QVector3D pos);
    void removeCellIfPresent (QVector3D pos, Cell* cell);
    Cell* getCell (QVector3D pos) const;
    Cell* getCellFast (const int& x, const int& y) const;

    //location functions
    QSet< CellCluster* > getNearbyClusters (const QVector3D& pos, qreal r) const;
    CellCluster* getNearbyClusterFast (const QVector3D& pos, qreal r, qreal minMass, qreal maxMass, CellCluster* exclude) const;
    using CellSelectFunction = bool(*)(Cell*);
    QList< Cell* > getNearbySpecificCells (const QVector3D& pos, qreal r, CellSelectFunction selection) const;

    //energy grid access functions
    void removeEnergy (QVector3D pos, EnergyParticle* energy);
    EnergyParticle* getEnergyFast (const int& x, const int& y) const;
    void setEnergy(QVector3D pos, EnergyParticle* energy);
    EnergyParticle* getEnergy (QVector3D pos) const;

    //auxiliary functions
    void correctPosition (QVector3D& pos) const;
    void correctDisplacement (QVector3D& displacement) const;
    QVector3D displacement (QVector3D fromPoint, QVector3D toPoint) const;
    QVector3D displacement (Cell* fromCell, Cell* toCell) const;
    qreal distance (Cell* fromCell, Cell* toCell) const;

    //(de)serialisation functions
    void serializeSize (QDataStream& stream) const;
    void serializeMap (QDataStream& stream) const;
    void buildEmptyMap (QDataStream& stream);
    void buildMap (QDataStream& stream, const QMap< quint64, Cell* >& oldIdCellMap, const QMap< quint64, EnergyParticle* >& oldIdEnergyMap);

private:
    QMutex _mutex;
    qint32 _sizeX;
    qint32 _sizeY;
    Cell*** _cellGrid;
    EnergyParticle*** _energyGrid;

    QList< CellCluster* > _clusters;
    QList< EnergyParticle* > _energyParticles;
};

/******************
 * inline functions
 ******************/
inline QList< CellCluster* >& Grid::getClusters ()
{
    return _clusters;
}

inline QList< EnergyParticle* >& Grid::getEnergyParticles ()
{
    return _energyParticles;
}

inline void Grid::setCell (QVector3D pos, Cell* cell)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    _cellGrid[x][y] = cell;
}

inline Cell* Grid::getCell (QVector3D pos) const
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    return _cellGrid[x][y];
}

inline Cell* Grid::getCellFast (const int &x, const int& y) const
{
    return _cellGrid[x][y];
}

inline EnergyParticle* Grid::getEnergyFast (const int& x, const int& y) const
{
    return _energyGrid[x][y];
}

inline QSet< CellCluster* > Grid::getNearbyClusters (const QVector3D& pos, qreal r) const
{
    QSet< CellCluster* > clusters;
//    int r = qFloor(simulationParameters.CRIT_CELL_DIST_MAX+1.0);
    int rc = qCeil(r);
    for(int rx = pos.x()-rc; rx < pos.x()+rc+1; ++rx)
        for(int ry = pos.y()-rc; ry < pos.y()+rc+1; ++ry) {
            if( QVector3D(static_cast<float>(rx)-pos.x(),static_cast<float>(ry)-pos.y(),0).length() < r+ALIEN_PRECISION ) {
                Cell* cell(getCell(QVector3D(rx,ry,0)));
                if( cell )
                    clusters << cell->getCluster();
            }
        }
    return clusters;
}

inline QList< Cell* > Grid::getNearbySpecificCells (const QVector3D& pos, qreal r, CellSelectFunction selection) const
{
    QList< Cell* > cells;
    int rCeil = qCeil(r);
    for(int scanX = pos.x()-rCeil; scanX < pos.x()+rCeil+1; ++scanX)
        for(int scanY = pos.y()-rCeil; scanY < pos.y()+rCeil+1; ++scanY) {
            if( QVector3D(static_cast<float>(scanX)-pos.x(),static_cast<float>(scanY)-pos.y(),0).length() < r+ALIEN_PRECISION ) {
                Cell* cell(getCell(QVector3D(scanX, scanY,0)));
                if( cell ) {
                    if( selection(cell) ) {
                        cells << cell;
                    }
                }
            }
        }
    return cells;
}

inline void Grid::removeEnergy (QVector3D pos, EnergyParticle* energy)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    if( _energyGrid[x][y] == energy )
        _energyGrid[x][y] = 0;
}

inline void Grid::setEnergy (QVector3D pos, EnergyParticle* energy)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    _energyGrid[x][y] = energy;
}

inline EnergyParticle* Grid::getEnergy (QVector3D pos) const
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    return _energyGrid[x][y];
}

#endif // GRID_H
