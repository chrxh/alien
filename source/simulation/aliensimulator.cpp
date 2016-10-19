#include "aliensimulator.h"

#include "processing/alienthread.h"
#include "entities/aliengrid.h"
#include "entities/aliencellcluster.h"
#include "processing/aliencellfunction.h"
#include "processing/aliencellfunctionfactory.h"
#include "physics/physics.h"
#include "../globaldata/simulationsettings.h"
#include "../globaldata/globalfunctions.h"
#include "metadatamanager.h"

#include <QTimer>
#include <QtCore/qmath.h>

AlienSimulator::AlienSimulator(int sizeX, int sizeY, QObject* parent)
    : QObject(parent), _run(false), _fps(0), _calculating(false), _frame(0), _newCellTokenAccessNumber(0)
{
    _forceFpsTimer = new QTimer(this);
    _grid = new AlienGrid(this);
    _thread = new AlienThread();

    connect(_forceFpsTimer, SIGNAL(timeout()), this, SLOT(forceFpsTimerSlot()));
    connect(this, SIGNAL(setRandomSeed(uint)), _thread, SLOT(setRandomSeed(uint)));
    connect(this, SIGNAL(calcNextTimestep()), _thread, SLOT(calcNextTimestep()));
    connect(_thread, SIGNAL(nextTimestepCalculated()), this, SLOT(nextTimestepCalculated()));

    //init simulation
    _grid->init(sizeX, sizeY);
    _thread->init(_grid);

    //start thread
    _thread->start();
    _thread->moveToThread(_thread);
    emit setRandomSeed(0);
}

AlienSimulator::~AlienSimulator ()
{
    _thread->quit();
    if( !_thread->wait(2000) ) {
        _thread->terminate();
        _thread->wait();
    }
    delete _thread;
}

QMap< QString, qreal > AlienSimulator::getMonitorData ()
{
    QMap< QString, qreal > data;
    _grid->lockData();
    int cells(0);
    int particles(0);
    int token(0);
    qreal internalEnergy(_thread->calcInternalEnergy());
    foreach( AlienCellCluster* cluster, _thread->getClusters() ) {
        cells += cluster->getCells().size();
        foreach( AlienCell* cell, cluster->getCells() ) {
            token += cell->getNumToken();
        }
    }
    particles = _thread->getEnergyParticles().size();
    _grid->unlockData();
    data["cells"] = cells;
    data["clusters"] = _thread->getClusters().size();
    data["energyParticles"] = particles;
    data["token"] = token;
    data["internalEnergy"] = internalEnergy;
    data["transEnergy"] = _thread->calcTransEnergy()/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
    data["rotEnergy"] = _thread->calcRotEnergy()/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
    return data;
}

void AlienSimulator::newUniverse (qint32 sizeX, qint32 sizeY)
{
    _grid->lockData();
    _frame = 0;

    //clean up metadata
    QSet< quint64 > ids = _grid->getAllCellIds();
    MetadataManager::getGlobalInstance().cleanUp(ids);

    //set up new grid
    _grid->reinit(sizeX, sizeY);

    _grid->unlockData();
}

void AlienSimulator::serializeUniverse (QDataStream& stream)
{
    //reset random seed for simulation thread to be deterministic
    emit setRandomSeed(_frame);

    _grid->lockData();
    stream << _frame;

    //clean up metadata
    QSet< quint64 > ids = _grid->getAllCellIds();
    MetadataManager::getGlobalInstance().cleanUp(ids);

    //serialize grid size
    _grid->serializeSize(stream);

    //serialize clusters
    quint32 numCluster = _thread->getClusters().size();
    stream << numCluster;
    foreach(AlienCellCluster* cluster, _thread->getClusters())
        cluster->serialize(stream);

    //serialize energy particles
    quint32 numEnergyParticles = _thread->getEnergyParticles().size();
    stream << numEnergyParticles;
    foreach(AlienEnergy* e, _thread->getEnergyParticles())
        e->serialize(stream);

    //serialize map data
    _grid->serializeMap(stream);

    _grid->unlockData();
}

void AlienSimulator::buildUniverse (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap, QMap< quint64, quint64 >& oldNewCellIdMap)
{
    _grid->lockData();
    stream >> _frame;

    //maps for associating new cells and energy particles
    QMap< quint64, AlienCell* > oldIdCellMap;
    QMap< quint64, AlienEnergy* > oldIdEnergyMap;

    //construct empty map
    _grid->buildEmptyMap(stream);

    //reconstruct cluster
    quint32 numCluster;
    stream >> numCluster;
    for(auto i = 0; i < numCluster; ++i) {
        AlienCellCluster* cluster = AlienCellCluster::buildCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, _grid);
        _grid->getClusters() << cluster;
    }

    //reconstruct energy particles
    quint32 numEnergyParticles;
    stream >> numEnergyParticles;
    for(auto i = 0; i < numEnergyParticles; ++i) {
        AlienEnergy* e = new AlienEnergy(stream, oldIdEnergyMap, _grid);
        _grid->getEnergyParticles() << e;
    }

    //reconstruct map
    _grid->buildMap(stream, oldIdCellMap, oldIdEnergyMap);

    _grid->unlockData();

    //reset random seed for simulation thread to be deterministic
    emit setRandomSeed(_frame);
}

qint32 AlienSimulator::getUniverseSizeX ()
{
    _grid->lockData();
    quint32 sizeX = _grid->getSizeX();
    _grid->unlockData();
    return sizeX;
}

qint32 AlienSimulator::getUniverseSizeY ()
{
    _grid->lockData();
    quint32 sizeY = _grid->getSizeY();
    _grid->unlockData();
    return sizeY;
}

void AlienSimulator::addBlockStructure (QVector3D center, int numCellX, int numCellY, QVector3D dist, qreal energy)
{
    //create cell grid
    AlienCell* cellGrid[numCellX][numCellY];
    for(int i = 0; i < numCellX; ++i )
        for(int j = 0; j < numCellY; ++j ) {
            qreal x = - ((qreal)numCellX-1.0)*dist.x()/2.0 + (qreal)i*dist.x();
            qreal y = - ((qreal)numCellY-1.0)*dist.y()/2.0 + (qreal)j*dist.y();
            int maxCon = 4;
            if( (i == 0) || (i == (numCellX-1)) || (j == 0) || (j == (numCellY-1)) )
                maxCon = 3;
            if( ((i == 0) || (i == (numCellX-1))) && ((j == 0) || (j == (numCellY-1))) )
                maxCon = 2;
            AlienCell* cell = AlienCell::buildCell(energy, _grid, maxCon, 0, 0, QVector3D(x, y, 0.0));

            cellGrid[i][j] = cell;
        }
    QList< AlienCell* > cells;
    for(int i = 0; i < numCellX; ++i )
        for(int j = 0; j < numCellY; ++j ) {
            if( i < (numCellX-1) )
                cellGrid[i][j]->newConnection(cellGrid[i+1][j]);
            if( j < (numCellY-1) )
                cellGrid[i][j]->newConnection(cellGrid[i][j+1]);
            cells << cellGrid[i][j];
        }

    //create cluster
    _grid->lockData();
    AlienCellCluster* cluster = AlienCellCluster::buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _grid);
    cluster->drawCellsToMap();
    _thread->getClusters() << cluster;
    _grid->unlockData();
    QList< AlienCellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void AlienSimulator::addHexagonStructure (QVector3D center, int numLayers, qreal dist, qreal energy)
{
    //create hexagon cell structure
    AlienCell* cellGrid[2*numLayers-1][2*numLayers-1];
    QList< AlienCell* > cells;
    int maxCon = 6;
    qreal incY = qSqrt(3)*dist/2.0;
    for(int j = 0; j < numLayers; ++j) {
        for(int i = -(numLayers-1); i < numLayers-j; ++i) {

            //check if cell is on boundary
            if( ((i == -(numLayers-1)) || (i == numLayers-j-1)) && ((j == 0) || (j == numLayers-1)) )
                maxCon = 3;
            else if( (i == -(numLayers-1)) || (i == numLayers-j-1) || (j == numLayers-1) )
                maxCon = 4;
            else
                maxCon = 6;

            //create cell: upper layer
            cellGrid[numLayers-1+i][numLayers-1-j] = AlienCell::buildCell(energy, _grid, maxCon, 0, 0, QVector3D(i*dist+j*dist/2.0, -j*incY, 0.0));
            cells << cellGrid[numLayers-1+i][numLayers-1-j];
            if( numLayers-1+i > 0 )
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i-1][numLayers-1-j]);
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i][numLayers-1-j+1]);
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i+1][numLayers-1-j+1]);
            }

            //create cell: under layer (except for 0-layer)
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1+j] = AlienCell::buildCell(energy, _grid, maxCon, 0, 0, QVector3D(i*dist+j*dist/2.0, +j*incY, 0.0));
                cells << cellGrid[numLayers-1+i][numLayers-1+j];
                if( numLayers-1+i > 0 )
                    cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i-1][numLayers-1+j]);
                cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i][numLayers-1+j-1]);
                cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i+1][numLayers-1+j-1]);
            }
        }
    }

    //create cluster
    _grid->lockData();
    AlienCellCluster* cluster = AlienCellCluster::buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _grid);
    cluster->drawCellsToMap();
    _thread->getClusters() << cluster;
    _grid->unlockData();
    QList< AlienCellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void AlienSimulator::addRandomEnergy (qreal energy, qreal maxEnergyPerParticle)
{
    _grid->lockData();

    while( energy > 0.0 ) {
        qreal rEnergy = GlobalFunctions::random(0.01, maxEnergyPerParticle);
        if( rEnergy > energy )
            rEnergy = energy;
        qreal posX = GlobalFunctions::random(0.0, _grid->getSizeX());
        qreal posY = GlobalFunctions::random(0.0, _grid->getSizeY());
        qreal velX = GlobalFunctions::random(-1.0, 1.0);
        qreal velY = GlobalFunctions::random(-1.0, 1.0);

        AlienEnergy* e = new AlienEnergy(rEnergy, QVector3D(posX, posY, 0.0), QVector3D(velX, velY, 0.0), _grid);
        _grid->getEnergyParticles() << e;
        _grid->setEnergy(e->pos, e);
        energy -= rEnergy;
    }
    _grid->unlockData();
}

void AlienSimulator::serializeCell (QDataStream& stream, AlienCell* cell, quint64& clusterId, quint64& cellId)
{
    _grid->lockData();

    //serialize cell data
    cell->serialize(stream);
    stream << clusterId;

    //get ids
    AlienCellCluster* cluster = cell->getCluster();
    clusterId = cluster->getId();
    cellId = cell->getId();

    _grid->unlockData();
}

void AlienSimulator::serializeExtendedSelection (QDataStream& stream, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es, QList< quint64 >& clusterIds, QList< quint64 >& cellIds)
{
    _grid->lockData();

    //serialize num clusters
    quint32 numClusters = clusters.size();
    stream << numClusters;

    //serialize cluster data
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->serialize(stream);
        clusterIds << cluster->getId();
        foreach(AlienCell* cell, cluster->getCells())
            cellIds << cell->getId();
    }

    //serialize num energy particles
    quint32 numEs = es.size();
    stream << numEs;

    //serialize energy particle data
    foreach(AlienEnergy* e, es) {
        e->serialize(stream);
    }

    _grid->unlockData();
}

void AlienSimulator::buildCell (QDataStream& stream,
                QVector3D pos,
                AlienCellCluster*& newCluster,
                QMap< quint64, quint64 >& oldNewClusterIdMap,
                QMap< quint64, quint64 >& oldNewCellIdMap,
                bool drawToMap)
{
    _grid->lockData();

    //read cell data
    QList< AlienCell* > newCells;
    AlienCell* newCell = AlienCell::buildCellWithoutConnectingCells(stream, _grid);
    newCells << newCell;
    newCells[0]->setRelPos(QVector3D());
    newCluster = AlienCellCluster::buildCellCluster(newCells, 0, pos, 0, newCell->getVel(), _grid);
    _thread->getClusters() << newCluster;

    //read old cluster id
    quint64 oldClusterId(0);
    stream >> oldClusterId;

    //assigning new ids
    newCluster->setId(GlobalFunctions::getTag());
    oldNewClusterIdMap[oldClusterId] = newCluster->getId();
    quint64 oldCellId = newCell->getId();
    newCell->setId(GlobalFunctions::getTag());
    oldNewCellIdMap[oldCellId] = newCell->getId();

    //draw cluster
    if( drawToMap )
        newCluster->drawCellsToMap();

    _grid->unlockData();
}

void AlienSimulator::buildExtendedSelection (QDataStream& stream,
                                             QVector3D pos,
                                             QList< AlienCellCluster* >& newClusters,
                                             QList< AlienEnergy* >& newEnergyParticles,
                                             QMap< quint64, quint64 >& oldNewClusterIdMap,
                                             QMap< quint64, quint64 >& oldNewCellIdMap,
                                             bool drawToMap)
{
    _grid->lockData();

    //maps for associating new cells and energy particles
    QMap< quint64, AlienCell* > oldIdCellMap;
    QMap< quint64, AlienEnergy* > oldIdEnergyMap;

    //read num clusters
    quint32 numClusters;
    stream >> numClusters;

    //read cluster data
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;
    for(int i = 0; i < numClusters; ++i) {
        AlienCellCluster* cluster = AlienCellCluster::buildCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, _grid);
        newClusters << cluster;
        foreach(AlienCell* cell, cluster->getCells())
            center += cell->calcPosition();
        numCells += cluster->getCells().size();
    }
    _thread->getClusters() << newClusters;

    //read num energy particles
    quint32 numEnergyParticles;
    stream >> numEnergyParticles;

    //read energy particle data
    for(int i = 0; i < numEnergyParticles; ++i) {
        AlienEnergy* e(new AlienEnergy(stream, oldIdEnergyMap, _grid));
        center += e->pos;
        newEnergyParticles << e;
    }
    _grid->getEnergyParticles() << newEnergyParticles;

    //set new center and draw cluster
    center = center / (qreal)(numCells+numEnergyParticles);
    foreach(AlienCellCluster* cluster, newClusters) {
        cluster->setPosition(cluster->getPosition()-center+pos);
        cluster->calcTransform();
        if( drawToMap )
            cluster->drawCellsToMap();
    }
    foreach(AlienEnergy* e, newEnergyParticles) {
        e->pos = e->pos-center+pos;
        if( drawToMap )
            _grid->setEnergy(e->pos, e);
    }
    _grid->unlockData();
}

void AlienSimulator::delSelection (QList< AlienCell* > cells, QList< AlienEnergy* > es)
{
    _grid->lockData();

    //remove energy particles
    foreach(AlienEnergy* e, es) {
        _thread->getEnergyParticles().removeAll(e);
        _grid->removeEnergy(e->pos, e);
        delete e;
    }

    //remove cells
    QList< AlienCellCluster* > allNewClusters;
    foreach(AlienCell* cell, cells) {

        //remove cell from cluster structure
        AlienCellCluster* cluster = cell->getCluster();
        _grid->removeCellIfPresent(cluster->calcPosition(cell, _grid), cell);
        cluster->removeCell(cell, false);
        delete cell;

        //calculate new cluster structure
        QList< AlienCellCluster* > newClusters;
        newClusters = cluster->decompose();
        _thread->getClusters() << newClusters;
        _thread->getClusters().removeAll(cluster);
        allNewClusters.removeAll(cluster);
        allNewClusters << newClusters;
        delete cluster;
    }

    //clean up metadata
//    QSet< quint64 > ids = _grid->getAllCellIds();
//    _meta->cleanUp(ids);

    _grid->unlockData();
//????
//    if( !allNewClusters.isEmpty() )
        emit reclustered(allNewClusters);
}

void AlienSimulator::delExtendedSelection (QList< AlienCellCluster* > clusters, QList< AlienEnergy* > es)
{
    _grid->lockData();

    //remove cell clusters
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->clearCellsFromMap();
        _thread->getClusters().removeAll(cluster);
        delete cluster;
    }

    //remove energy particles
    foreach(AlienEnergy* e, es) {
        _thread->getEnergyParticles().removeAll(e);
        _grid->removeEnergy(e->pos, e);
        delete e;
    }

    //clean up metadata
//    QSet< quint64 > ids = _grid->getAllCellIds();
//    _meta->cleanUp(ids);

    _grid->unlockData();
}

void AlienSimulator::rotateExtendedSelection (qreal angle, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es)
{
    _grid->lockData();

    //1. step: rotate each cluster around own center
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->setAngle(cluster->getAngle()+angle);
    }

    //2. step: rotate cluster around common center
    _grid->unlockData();
    QVector3D center = getCenterPosExtendedSelection(clusters, es);
    _grid->lockData();
    QMatrix4x4 transform;
    transform.setToIdentity();
    transform.translate(center);
    transform.rotate(angle, 0.0, 0.0, 1.0);
    transform.translate(-center);
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->setPosition(transform.map(cluster->getPosition()));
    }
    foreach(AlienEnergy* e, es) {
        e->pos = transform.map(e->pos);
    }
    _grid->unlockData();
}

void AlienSimulator::setVelocityXExtendedSelection (qreal velX, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es)
{
    _grid->lockData();
    foreach(AlienCellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVel();
        vel.setX(velX);
        cluster->setVel(vel);
    }
    foreach(AlienEnergy* e, es) {
        e->vel.setX(velX);
    }
    _grid->unlockData();
}

void AlienSimulator::setVelocityYExtendedSelection (qreal velY, const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es)
{
    _grid->lockData();
    foreach(AlienCellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVel();
        vel.setY(velY);
        cluster->setVel(vel);
    }
    foreach(AlienEnergy* e, es) {
        e->vel.setY(velY);
    }
    _grid->unlockData();
}

void AlienSimulator::setAngularVelocityExtendedSelection (qreal angVel, const QList< AlienCellCluster* >& clusters)
{
    _grid->lockData();
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->setAngularVel(angVel);
    }
    _grid->unlockData();
}


QVector3D AlienSimulator::getCenterPosExtendedSelection (const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es)
{
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;

    _grid->lockData();
    foreach(AlienCellCluster* cluster, clusters) {
        center += cluster->getPosition()*cluster->getMass();
/*        foreach(AlienCell* cell, cluster->getCells())
            center += cell->calcPosition();*/
        numCells += cluster->getCells().size();
    }
    foreach(AlienEnergy* e, es) {
        center += e->pos;
    }
    _grid->unlockData();

    center = center / (qreal)(numCells+es.size());
    return center;
}

void AlienSimulator::drawToMapExtendedSelection (const QList< AlienCellCluster* >& clusters, const QList< AlienEnergy* >& es)
{
    _grid->lockData();
    foreach(AlienCellCluster* cluster, clusters) {
        cluster->drawCellsToMap();
    }
    foreach(AlienEnergy* e, es) {
        _grid->setEnergy(e->pos, e);
    }
    _grid->unlockData();
}

void AlienSimulator::newCell (QVector3D pos)
{
    //create cluster with single cell
    _grid->lockData();
    AlienCell* cell = AlienCell::buildCell(simulationParameters.NEW_CELL_ENERGY,
                                           _grid,
                                           simulationParameters.NEW_CELL_MAX_CONNECTION,
                                           simulationParameters.NEW_CELL_TOKEN_ACCESS_NUMBER);
    cell->setTokenAccessNumber(_newCellTokenAccessNumber++);
    QList< AlienCell* > cells;
    cells << cell;
    AlienCellCluster* cluster = AlienCellCluster::buildCellCluster(cells, 0.0, pos, 0, QVector3D(), _grid);
    _grid->setCell(pos, cell);
    _thread->getClusters() << cluster;
    _grid->unlockData();

    emit cellCreated(cell);
}

void AlienSimulator::newEnergyParticle (QVector3D pos)
{
    //create energy particle
    _grid->lockData();
    AlienEnergy* energy = new AlienEnergy(simulationParameters.CRIT_CELL_TRANSFORM_ENERGY/2, pos, QVector3D(), _grid);
    _grid->setEnergy(pos, energy);
    _thread->getEnergyParticles() << energy;
    _grid->unlockData();

    emit energyParticleCreated(energy);
}


void AlienSimulator::updateCell (QList< AlienCell* > cells, QList< AlienCellReduced > newCellsData, bool clusterDataChanged)
{
    //update purely cell data and no cluster data?
    if( !clusterDataChanged ) {
        _grid->lockData();

        QListIterator< AlienCell* > iCells(cells);
        QListIterator< AlienCellReduced > iNewCellsData(newCellsData);
        QSet< AlienCellCluster* > sumNewClusters;
        while (iCells.hasNext()) {

            AlienCell* cell = iCells.next();
            AlienCellReduced newCellData = iNewCellsData.next();

            //update cell properties
            cell->getCluster()->calcTransform();
            cell->setAbsPositionAndUpdateMap(newCellData.cellPos);
            cell->setEnergy(newCellData.cellEnergy);
            cell->delAllConnection();
            if( newCellData.cellMaxCon > simulationParameters.MAX_CELL_CONNECTIONS )
                newCellData.cellMaxCon = simulationParameters.MAX_CELL_CONNECTIONS;
            cell->setMaxConnections(newCellData.cellMaxCon);
            cell->setBlockToken(!newCellData.cellAllowToken);
            cell->setTokenAccessNumber(newCellData.cellTokenAccessNum);
            cell->setCellFunction(AlienCellFunctionFactory::build(newCellData.cellFunctionName, false, _grid));

            //update cell computer
            for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
                cell->getMemory()[i] = newCellData.computerMemory[i];
            int errorLine = 0;
            if( cell->getCellFunction()->compileCode(newCellData.computerCode, errorLine) == false )
                emit computerCompilationReturn(true, errorLine);
            else
                emit computerCompilationReturn(false, 0);


            //update token
            cell->delAllTokens();
            int numToken = newCellData.tokenEnergies.size();
            for( int i = 0; i < numToken; ++i )
                cell->addToken(new AlienToken(newCellData.tokenEnergies[i], newCellData.tokenData[i]), true, false);

            //searching for nearby clusters
            QVector3D pos = cell->calcPosition();
            QSet< AlienCellCluster* > clusters(_grid->getNearbyClusters(pos, qFloor(simulationParameters.CRIT_CELL_DIST_MAX+1.0)));
//            if( !clusters.contains(cell->getCluster()) )
            clusters << cell->getCluster();

            //update cell velocities
/*            foreach(AlienCellCluster* cluster, clusters) {
                cluster->updateCellVel();
            }
*/
            //searching for nearby cells
            QList< AlienCell* > neighborCells;
            QList< QVector3D > neighborCellsRelPos;
            foreach(AlienCellCluster* cluster, clusters)
                foreach(AlienCell* otherCell, cluster->getCells()) {
                    QVector3D displacement = otherCell->calcPosition()-pos;
                    _grid->correctDisplacement(displacement);
                    qreal dist = displacement.length();
                    if( (cell != otherCell) && (dist < simulationParameters.CRIT_CELL_DIST_MAX) ) {

                        //cells connectable?
                        if( cell->connectable(otherCell)) {
                            neighborCells << otherCell;
                            neighborCellsRelPos << displacement;
                        }
                    }
                }


            //establishing new connections
            while( (!neighborCells.isEmpty()) && (cell->getNumConnections() < cell->getMaxConnections()) ) {
                QListIterator< QVector3D > it(neighborCellsRelPos);
                qreal min = 0.0;
                int index = 0;
                int minIndex = 0;
                while( it.hasNext() ) {
                    qreal dist(it.next().length());
                    if( (min == 0.0) || (dist <= min) ) {
                        min = dist;
                        minIndex = index;
                    }
                    index++;
                }
                AlienCell* minCell = neighborCells.at(minIndex);
                neighborCells.removeAt(minIndex);
                neighborCellsRelPos.removeAt(minIndex);

                //still possible to make connection between cell and minCell?
                if( cell->connectable(minCell)) {

                    //make new connection
                    cell->newConnection(minCell);
                }
            }

            //calculating new cell clusters
            QList< AlienCellCluster* > newClusters;
            foreach(AlienCellCluster* cluster, clusters){//_thread->getClusters()) {
                newClusters << cluster->decompose();
                _thread->getClusters().removeAll(cluster);
                sumNewClusters.remove(cluster);
                delete cluster;
            }
            _thread->getClusters() << newClusters;

    //        if( _grid->getCell(cell->calcPosition()) == 0 )
    //            _grid->setCell(cell->calcPosition(), cell);
            sumNewClusters.unite(newClusters.toSet());
        }

        //update cell velocities
        QList< AlienCellCluster* > changedClusters = sumNewClusters.toList();
        foreach(AlienCellCluster* cluster, changedClusters)
            cluster->updateCellVel();

        _grid->unlockData();

        //inform other instances about reclustering
        emit reclustered(changedClusters);
    }
    //update only cluster data
    else {
        _grid->lockData();

        QListIterator< AlienCell* > iCells(cells);
        QListIterator< AlienCellReduced > iNewCellsData(newCellsData);
        QList< AlienCellCluster* > sumNewClusters;
        while (iCells.hasNext()) {
            AlienCell* cell = iCells.next();
            AlienCellReduced newCellData = iNewCellsData.next();
            AlienCellCluster* cluster = cell->getCluster();
            cluster->clearCellsFromMap();
            cluster->setPosition(newCellData.clusterPos);
            cluster->setAngle(newCellData.clusterAngle);
            cluster->setVel(newCellData.clusterVel);
            cluster->setAngularVel(newCellData.clusterAngVel);
            cluster->calcTransform();
            cluster->drawCellsToMap();
            cluster->updateCellVel();
/*            foreach (AlienCell* otherCell, cluster->getCells() ) {
                otherCell->setVel(QVector3D());
            }*/
            sumNewClusters<< cluster;
        }
        _grid->unlockData();
        emit reclustered(sumNewClusters);
    }
}

void AlienSimulator::setRun (bool run)
{
    _run = run;
    if( run ) {
        _calculating = true;
        emit calcNextTimestep();
    }

    //stop play => inform other corredinators about actual state
    else
        updateUniverse();
}

//fps = 0: deactivate forcing
void AlienSimulator::forceFps (int fps)
{
    _fps = fps;
    _forceFpsTimer->stop();
    if( fps > 0 ){
        _forceFpsTimer->start(1000/fps);
    }
    else {
        if( _run && (!_calculating) ) {
            _calculating = true;
            emit calcNextTimestep();
        }
    }
}

void AlienSimulator::requestNextTimestep ()
{
    emit calcNextTimestep();
}

void AlienSimulator::updateUniverse ()
{
    emit universeUpdated(_grid, true);
}

void AlienSimulator::forceFpsTimerSlot ()
{
    if( _run ) {
        if( !_calculating ) {
            _calculating = true;
            emit calcNextTimestep();
        }
    }
}

void AlienSimulator::nextTimestepCalculated ()
{
    _frame++;
    _calculating = false;
    emit universeUpdated(_grid, false);

    //fps forcing must be deactivate in order to continue with next the frame
    if( _run ) {
        if( _fps == 0 )
            emit calcNextTimestep();
    }
}

