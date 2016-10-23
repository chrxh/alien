#include "aliengrid.h"
#include "aliencell.h"
#include "aliencellcluster.h"
#include "alienenergy.h"

#include "global/simulationsettings.h"

#include <QMutex>
#include <cmath>

AlienGrid::AlienGrid(QObject* parent)
    : QObject(parent), _cellGrid(0), _energyGrid(0)
{
}


AlienGrid::~AlienGrid ()
{
    foreach( AlienCellCluster* cluster, _clusters) {
        delete cluster;
    }
    foreach( AlienEnergy* energy, _energyParticles) {
        delete energy;
    }

    if( _cellGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _cellGrid[x];
        }
        delete [] _cellGrid;
    }
    if( _energyGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _energyGrid[x];
        }
        delete [] _energyGrid;
    }
}

void AlienGrid::init (qint32 sizeX, qint32 sizeY)
{
    _sizeX = sizeX;
    _sizeY = sizeY;

    _cellGrid = new AlienCell**[sizeX];
    for( qint32 x = 0; x < sizeX; ++x ) {
        _cellGrid[x] = new AlienCell*[sizeY];
    }
    _energyGrid = new AlienEnergy**[sizeX];
    for( qint32 x = 0; x < sizeX; ++x ) {
        _energyGrid[x] = new AlienEnergy*[sizeY];
    }
    clearGrids();
}

void AlienGrid::reinit (qint32 sizeX, qint32 sizeY)
{
    foreach( AlienCellCluster* cluster, _clusters) {
        delete cluster;
    }
    foreach( AlienEnergy* energy, _energyParticles) {
        delete energy;
    }
    _clusters.clear();
    _energyParticles.clear();

    if( _cellGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _cellGrid[x];
        }
        delete [] _cellGrid;
    }
    if( _energyGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _energyGrid[x];
        }
        delete [] _energyGrid;
    }

    init(sizeX, sizeY);
}

void AlienGrid::lockData ()
{
    _mutex.lock();
}

void AlienGrid::unlockData ()
{
    _mutex.unlock();
}

QSet< quint64 > AlienGrid::getAllCellIds () const
{
    QList< quint64 > cellIds;
    foreach(AlienCellCluster* cluster, _clusters) {
        cellIds << cluster->getCellIds();
    }
    return cellIds.toSet();
}

void AlienGrid::clearGrids ()
{
    for(qint32 x=0; x < _sizeX; ++x)
        for(qint32 y=0; y < _sizeY; ++y) {
            _cellGrid[x][y] = 0;
            _energyGrid[x][y] = 0;
        }
}

qint32 AlienGrid::getSizeX() const
{
    return _sizeX;
}

qint32 AlienGrid::getSizeY() const
{
    return _sizeY;
}


void AlienGrid::removeCell (QVector3D pos)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX-1)%_sizeX;
    y = ((y%_sizeY)+_sizeY-1)%_sizeY;

    _cellGrid[x][y] = 0;
    _cellGrid[(x+1)%_sizeX][y] = 0;
    _cellGrid[(x+2)%_sizeX][y] = 0;

    _cellGrid[x][(y+1)%_sizeY] = 0;
    _cellGrid[(x+1)%_sizeX][(y+1)%_sizeY] = 0;
    _cellGrid[(x+2)%_sizeX][(y+1)%_sizeY] = 0;

    _cellGrid[x][(y+2)%_sizeY] = 0;
    _cellGrid[(x+1)%_sizeX][(y+2)%_sizeY] = 0;
    _cellGrid[(x+2)%_sizeX][(y+2)%_sizeY] = 0;
}

void AlienGrid::removeCellIfPresent (QVector3D pos, AlienCell* cell)
{
    qint32 x = qFloor(pos.x());
    qint32 y = qFloor(pos.y());
    x = ((x%_sizeX)+_sizeX-1)%_sizeX;
    y = ((y%_sizeY)+_sizeY-1)%_sizeY;

    for(qint32 dx = 0; dx < 3; ++dx)
        for(qint32 dy = 0; dy < 3; ++dy) {
            if( _cellGrid[(x+dx)%_sizeX][(y+dy)%_sizeY] == cell) {
                _cellGrid[(x+dx)%_sizeX][(y+dy)%_sizeY] = 0;
            }
        }
}

AlienCellCluster* AlienGrid::getNearbyClusterFast (const QVector3D& pos, qreal r, qreal minMass, qreal maxMass, AlienCellCluster* exclude) const
{
    int step = qCeil(qSqrt(minMass+ALIEN_PRECISION))+3;  //horizontal or vertical length of cell cluster >= minDim
    int rc = qCeil(r);
    qreal rs = r*r+ALIEN_PRECISION;

    //grid scan
    AlienCellCluster* closestCluster = 0;
    qreal closestClusterDist = 0.0;
    qint32 x = qFloor(pos.x())-rc;
    qint32 y = qFloor(pos.y())-rc;
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    for(int rx = -rc; rx < rc; rx += step, x += step)
        for(int ry = -rc; ry < rc; ry += step, y += step) {
            if( static_cast<float>(rx*rx+ry*ry) < rs ) {
                x = x % _sizeX;
                y = y % _sizeY;
//                AlienCell* cell = getCell(QVector3D(x, y, 0.0));
                AlienCell* cell = getCellFast(x, y);
                if( cell ) {
                    AlienCellCluster* cluster = cell->getCluster();
                    if( cluster != exclude ) {

                        //compare masses
                        qreal mass = cluster->getMass();
                        if( mass >= (minMass-ALIEN_PRECISION) && mass <= (maxMass+ALIEN_PRECISION) ) {

                            //calc and compare dist
                            qreal dist = displacement(cell->calcPosition(), pos).length();
                            if( !closestCluster || (dist < closestClusterDist) ) {
                                closestCluster = cluster;
                                closestClusterDist = dist;
                            }
                        }
                    }
                }
            }
        }
    return closestCluster;
}

void AlienGrid::correctPosition (QVector3D& pos) const
{
    qint32 intPart(0.0);
    qreal fracPart(0.0);
    intPart = qFloor(pos.x());
    fracPart = pos.x()-intPart;
    pos.setX((qreal)(((intPart%_sizeX)+_sizeX)%_sizeX)+fracPart);
    intPart = qFloor(pos.y());
    fracPart = pos.y()-intPart;
    pos.setY((qreal)(((intPart%_sizeY)+_sizeY)%_sizeY)+fracPart);
}

void AlienGrid::correctDisplacement (QVector3D& displacement) const
{
    qint32 x = qFloor(displacement.x());
    qint32 y = qFloor(displacement.y());
    qreal rx = displacement.x()-(qreal)x;
    qreal ry = displacement.y()-(qreal)y;
    x += _sizeX/2;
    y += _sizeY/2;
    x = ((x%_sizeX)+_sizeX)%_sizeX;
    y = ((y%_sizeY)+_sizeY)%_sizeY;
    x -= _sizeX/2;
    y -= _sizeY/2;
    displacement.setX((qreal)x+rx);
    displacement.setY((qreal)y+ry);
}

QVector3D AlienGrid::displacement (QVector3D fromPoint, QVector3D toPoint) const
{
    QVector3D d = toPoint-fromPoint;
    correctDisplacement(d);
    return d;
}

QVector3D AlienGrid::displacement (AlienCell* fromCell, AlienCell* toCell) const
{
    return displacement(fromCell->calcPosition(), toCell->calcPosition());
}

qreal AlienGrid::distance (AlienCell* fromCell, AlienCell* toCell) const
{
    return displacement(fromCell, toCell).length();
}


void AlienGrid::serializeSize (QDataStream& stream) const
{
    stream << _sizeX << _sizeY;
}

void AlienGrid::serializeMap (QDataStream& stream) const
{
    //determine number of cell entries
    quint32 numEntries = 0;
    for(qint32 x = 0; x < _sizeX; ++x)
        for(qint32 y = 0; y < _sizeY; ++y)
            if( _cellGrid[x][y] )
                numEntries++;
    stream << numEntries;

    //write cell entries
    for(qint32 x = 0; x < _sizeX; ++x)
        for(qint32 y = 0; y < _sizeY; ++y) {
            AlienCell* cell = _cellGrid[x][y];
            if( cell ) {
                stream << x << y << cell->getId();
            }

        }

    //determine number of energy particle entries
    numEntries = 0;
    for(qint32 x = 0; x < _sizeX; ++x)
        for(qint32 y = 0; y < _sizeY; ++y)
            if( _energyGrid[x][y] )
                numEntries++;
    stream << numEntries;

    //write energy particle entries
    for(qint32 x = 0; x < _sizeX; ++x)
        for(qint32 y = 0; y < _sizeY; ++y) {
            AlienEnergy* e = _energyGrid[x][y];
            if( e ) {
                stream << x << y << e->id;
            }

        }
}

void AlienGrid::buildEmptyMap (QDataStream& stream)
{
    //delete clusters and energy particles
    foreach( AlienCellCluster* cluster, _clusters) {
        delete cluster;
    }
    foreach( AlienEnergy* energy, _energyParticles) {
        delete energy;
    }
    _clusters.clear();
    _energyParticles.clear();

    //delete old map
    if( _cellGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _cellGrid[x];
        }
        delete [] _cellGrid;
    }
    if( _energyGrid ) {
        for( qint32 x = 0; x < _sizeX; ++x ) {
            delete [] _energyGrid[x];
        }
        delete [] _energyGrid;
    }
    stream >> _sizeX >> _sizeY;

    //create new map
    _cellGrid = new AlienCell**[_sizeX];
    for( qint32 x = 0; x < _sizeX; ++x ) {
        _cellGrid[x] = new AlienCell*[_sizeY];
    }
    _energyGrid = new AlienEnergy**[_sizeX];
    for( qint32 x = 0; x < _sizeX; ++x ) {
        _energyGrid[x] = new AlienEnergy*[_sizeY];
    }
    clearGrids();
}

void AlienGrid::buildMap (QDataStream& stream, const QMap< quint64, AlienCell* >& oldIdCellMap, const QMap< quint64, AlienEnergy* >& oldIdEnergyMap)
{
    //read cell entries
    quint32 numEntries = 0;
    qint32 x = 0;
    qint32 y = 0;
    quint64 oldId = 0;
    stream >> numEntries;
    for(auto i = 0; i < numEntries; ++i) {
        stream >> x >> y >> oldId;
        _cellGrid[x][y] = oldIdCellMap[oldId];
    }

    //read energy particle entries
    numEntries = 0;
    stream >> numEntries;
    for(auto i = 0; i < numEntries; ++i) {
        stream >> x >> y >> oldId;
        _energyGrid[x][y] = oldIdEnergyMap[oldId];
    }
}



