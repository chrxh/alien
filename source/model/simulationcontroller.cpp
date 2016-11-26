#include "simulationcontroller.h"

#include "simulationunit.h"
#include "metadatamanager.h"
#include "factoryfacade.h"
#include "model/features/cellfunctioncomputer.h"
#include "model/entities/token.h"
#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/physics/physics.h"
#include "model/simulationsettings.h"
#include "global/global.h"
#include "global/servicelocator.h"

#include <QTimer>
#include <QtCore/qmath.h>
#include <QVector2D>
#include <QMatrix4x4>
#include <set>

SimulationController::SimulationController(Threading threading, QObject* parent)
    : QObject(parent)
{
    _forceFpsTimer = new QTimer(this);
    _grid = new Grid(this);
    _unit = new SimulationUnit();
    _unit->init(_grid);

    connect(&_thread, &QThread::finished, _unit, &QObject::deleteLater);
    connect(_forceFpsTimer, SIGNAL(timeout()), this, SLOT(forceFpsTimerSlot()));
    connect(this, SIGNAL(setRandomSeed(uint)), _unit, SLOT(setRandomSeed(uint)));

    if( threading == Threading::EXTRA_THREAD ) {
        connect(this, SIGNAL(calcNextTimestep()), _unit, SLOT(calcNextTimestep()));
        connect(_unit, SIGNAL(nextTimestepCalculated()), this, SLOT(nextTimestepCalculated()));

        //start thread
        _unit->moveToThread(&_thread);
        _thread.start();

    }
    if( threading == Threading::NO_EXTRA_THREAD ) {
        connect(this, SIGNAL(calcNextTimestep()), _unit, SLOT(calcNextTimestep()), Qt::DirectConnection);
        _unit->setParent(this);
    }

    emit setRandomSeed(0);
}

SimulationController::SimulationController(QVector2D size, Threading threading, QObject* parent)
    : SimulationController(threading, parent)
{
    _grid->init(size.x(), size.y());
}

SimulationController::~SimulationController ()
{
    _thread.quit();
    if( !_thread.wait(2000) ) {
        _thread.terminate();
        _thread.wait();
    }
}

QMap< QString, qreal > SimulationController::getMonitorData ()
{
    QMap< QString, qreal > data;
    _grid->lockData();
    int cells(0);
    int particles(0);
    int token(0);
    qreal internalEnergy(_unit->calcInternalEnergy());
    foreach( CellCluster* cluster, _unit->getClusters() ) {
        cells += cluster->getCellsRef().size();
        foreach( Cell* cell, cluster->getCellsRef() ) {
            token += cell->getNumToken();
        }
    }
    particles = _unit->getEnergyParticles().size();
    _grid->unlockData();
    data["cells"] = cells;
    data["clusters"] = _unit->getClusters().size();
    data["energyParticles"] = particles;
    data["token"] = token;
    data["internalEnergy"] = internalEnergy;
    data["transEnergy"] = _unit->calcTransEnergy()/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
    data["rotEnergy"] = _unit->calcRotEnergy()/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
    return data;
}

Grid* SimulationController::getGrid ()
{
    return _grid;
}

void SimulationController::newUniverse (qint32 sizeX, qint32 sizeY)
{
    _grid->lockData();
    _frame = 0;

    //clean up metadata
    std::set<quint64> ids = _grid->getAllCellIds();
    MetadataManager::getGlobalInstance().cleanUp(ids);

    //set up new grid
    _grid->reinit(sizeX, sizeY);

    _grid->unlockData();
}

void SimulationController::serializeUniverse (QDataStream& stream)
{
    //reset random seed for simulation thread to be deterministic
    emit setRandomSeed(_frame);

    _grid->lockData();
    stream << _frame;

    //clean up metadata
    std::set<quint64> ids = _grid->getAllCellIds();
    MetadataManager::getGlobalInstance().cleanUp(ids);

    //serialize grid size
    _grid->serializeSize(stream);

    //serialize clusters
    quint32 numCluster = _unit->getClusters().size();
    stream << numCluster;
    foreach(CellCluster* cluster, _unit->getClusters())
        cluster->serialize(stream);

    //serialize energy particles
    quint32 numEnergyParticles = _unit->getEnergyParticles().size();
    stream << numEnergyParticles;
    foreach(EnergyParticle* e, _unit->getEnergyParticles())
        e->serialize(stream);

    //serialize map data
    _grid->serializeMap(stream);

    _grid->unlockData();
}

void SimulationController::buildUniverse (QDataStream& stream)
{
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    _grid->lockData();
    stream >> _frame;

    //maps for associating new cells and energy particles
    QMap< quint64, Cell* > oldIdCellMap;
    QMap< quint64, EnergyParticle* > oldIdEnergyMap;

    //construct empty map
    _grid->buildEmptyMap(stream);

    //reconstruct cluster
    quint32 numCluster;
    stream >> numCluster;
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    for(quint32 i = 0; i < numCluster; ++i) {
        CellCluster* cluster = facade->buildCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, _grid);
        _grid->getClusters() << cluster;
    }

    //reconstruct energy particles
    quint32 numEnergyParticles;
    stream >> numEnergyParticles;
    for(quint32 i = 0; i < numEnergyParticles; ++i) {
        EnergyParticle* e = new EnergyParticle(stream, oldIdEnergyMap, _grid);
        _grid->getEnergyParticles() << e;
    }

    //reconstruct map
    _grid->buildMap(stream, oldIdCellMap, oldIdEnergyMap);

    _grid->unlockData();

    simulationParameters.readData(stream);
    MetadataManager::getGlobalInstance().readMetadataUniverse(stream, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readSymbolTable(stream);

    //reset random seed for simulation thread to be deterministic
    emit setRandomSeed(_frame);
}

qint32 SimulationController::getUniverseSizeX ()
{
    _grid->lockData();
    quint32 sizeX = _grid->getSizeX();
    _grid->unlockData();
    return sizeX;
}

qint32 SimulationController::getUniverseSizeY ()
{
    _grid->lockData();
    quint32 sizeY = _grid->getSizeY();
    _grid->unlockData();
    return sizeY;
}

void SimulationController::addBlockStructure (QVector3D center, int numCellX, int numCellY, QVector3D dist, qreal energy)
{
    //create cell grid
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    Cell* cellGrid[numCellX][numCellY];
    for(int i = 0; i < numCellX; ++i )
        for(int j = 0; j < numCellY; ++j ) {
            qreal x = - ((qreal)numCellX-1.0)*dist.x()/2.0 + (qreal)i*dist.x();
            qreal y = - ((qreal)numCellY-1.0)*dist.y()/2.0 + (qreal)j*dist.y();
            int maxCon = 4;
            if( (i == 0) || (i == (numCellX-1)) || (j == 0) || (j == (numCellY-1)) )
                maxCon = 3;
            if( ((i == 0) || (i == (numCellX-1))) && ((j == 0) || (j == (numCellY-1))) )
                maxCon = 2;
            Cell* cell = facade->buildFeaturedCell(energy, CellFunctionType::COMPUTER, _grid, maxCon, 0, QVector3D(x, y, 0.0));

            cellGrid[i][j] = cell;
        }
    QList< Cell* > cells;
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
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _grid);
    cluster->drawCellsToMap();
    _unit->getClusters() << cluster;
    _grid->unlockData();
    QList< CellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void SimulationController::addHexagonStructure (QVector3D center, int numLayers, qreal dist, qreal energy)
{
    //create hexagon cell structure
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    Cell* cellGrid[2*numLayers-1][2*numLayers-1];
    QList< Cell* > cells;
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
            cellGrid[numLayers-1+i][numLayers-1-j] = facade->buildFeaturedCell(energy, CellFunctionType::COMPUTER, _grid, maxCon, 0, QVector3D(i*dist+j*dist/2.0, -j*incY, 0.0));
            cells << cellGrid[numLayers-1+i][numLayers-1-j];
            if( numLayers-1+i > 0 )
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i-1][numLayers-1-j]);
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i][numLayers-1-j+1]);
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i+1][numLayers-1-j+1]);
            }

            //create cell: under layer (except for 0-layer)
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1+j] = facade->buildFeaturedCell(energy, CellFunctionType::COMPUTER, _grid, maxCon, 0, QVector3D(i*dist+j*dist/2.0, +j*incY, 0.0));
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
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _grid);
    cluster->drawCellsToMap();
    _unit->getClusters() << cluster;
    _grid->unlockData();
    QList< CellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void SimulationController::addRandomEnergy (qreal energy, qreal maxEnergyPerParticle)
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

        EnergyParticle* e = new EnergyParticle(rEnergy, QVector3D(posX, posY, 0.0), QVector3D(velX, velY, 0.0), _grid);
        _grid->getEnergyParticles() << e;
        _grid->setEnergy(e->pos, e);
        energy -= rEnergy;
    }
    _grid->unlockData();
}

void SimulationController::serializeCell (QDataStream& stream, Cell* cell, quint64& clusterId, quint64& cellId)
{
    _grid->lockData();

    //serialize cell data
    FactoryFacade *facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    facade->serializeFeaturedCell(cell, stream);
    stream << clusterId;

    //get ids
    CellCluster* cluster = cell->getCluster();
    clusterId = cluster->getId();
    cellId = cell->getId();

    _grid->unlockData();
}

void SimulationController::serializeExtendedSelection (QDataStream& stream, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es, QList< quint64 >& clusterIds, QList< quint64 >& cellIds)
{
    _grid->lockData();

    //serialize num clusters
    quint32 numClusters = clusters.size();
    stream << numClusters;

    //serialize cluster data
    foreach(CellCluster* cluster, clusters) {
        cluster->serialize(stream);
        clusterIds << cluster->getId();
        foreach(Cell* cell, cluster->getCellsRef())
            cellIds << cell->getId();
    }

    //serialize num energy particles
    quint32 numEs = es.size();
    stream << numEs;

    //serialize energy particle data
    foreach(EnergyParticle* e, es) {
        e->serialize(stream);
    }

    _grid->unlockData();
}

void SimulationController::buildCell (QDataStream& stream,
                QVector3D pos,
                CellCluster*& newCluster,
                QMap< quint64, quint64 >& oldNewClusterIdMap,
                QMap< quint64, quint64 >& oldNewCellIdMap,
                bool drawToMap)
{
    _grid->lockData();

    //read cell data
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    QList< Cell* > newCells;
    Cell* newCell = facade->buildFeaturedCell(stream, _grid);
    newCells << newCell;
    newCells[0]->setRelPos(QVector3D());
    newCluster = facade->buildCellCluster(newCells, 0, pos, 0, newCell->getVel(), _grid);
    _unit->getClusters() << newCluster;

    //read old cluster id
    quint64 oldClusterId(0);
    stream >> oldClusterId;

    //assigning new ids
    newCluster->setId(GlobalFunctions::createNewTag());
    oldNewClusterIdMap[oldClusterId] = newCluster->getId();
    quint64 oldCellId = newCell->getId();
    newCell->setId(GlobalFunctions::createNewTag());
    oldNewCellIdMap[oldCellId] = newCell->getId();

    //draw cluster
    if( drawToMap )
        newCluster->drawCellsToMap();

    _grid->unlockData();
}

void SimulationController::buildExtendedSelection (QDataStream& stream,
                                             QVector3D pos,
                                             QList< CellCluster* >& newClusters,
                                             QList< EnergyParticle* >& newEnergyParticles,
                                             QMap< quint64, quint64 >& oldNewClusterIdMap,
                                             QMap< quint64, quint64 >& oldNewCellIdMap,
                                             bool drawToMap)
{
    _grid->lockData();

    //maps for associating new cells and energy particles
    QMap< quint64, Cell* > oldIdCellMap;
    QMap< quint64, EnergyParticle* > oldIdEnergyMap;

    //read num clusters
    quint32 numClusters;
    stream >> numClusters;

    //read cluster data
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    for(int i = 0; i < numClusters; ++i) {
        CellCluster* cluster = facade->buildCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, _grid);
        newClusters << cluster;
        foreach(Cell* cell, cluster->getCellsRef())
            center += cell->calcPosition();
        numCells += cluster->getCellsRef().size();
    }
    _unit->getClusters() << newClusters;

    //read num energy particles
    quint32 numEnergyParticles;
    stream >> numEnergyParticles;

    //read energy particle data
    for(int i = 0; i < numEnergyParticles; ++i) {
        EnergyParticle* e(new EnergyParticle(stream, oldIdEnergyMap, _grid));
        center += e->pos;
        newEnergyParticles << e;
    }
    _grid->getEnergyParticles() << newEnergyParticles;

    //set new center and draw cluster
    center = center / (qreal)(numCells+numEnergyParticles);
    foreach(CellCluster* cluster, newClusters) {
        cluster->setPosition(cluster->getPosition()-center+pos);
        cluster->updateTransformationMatrix();
        if( drawToMap )
            cluster->drawCellsToMap();
    }
    foreach(EnergyParticle* e, newEnergyParticles) {
        e->pos = e->pos-center+pos;
        if( drawToMap )
            _grid->setEnergy(e->pos, e);
    }
    _grid->unlockData();
}

void SimulationController::delSelection (QList< Cell* > cells, QList< EnergyParticle* > es)
{
    _grid->lockData();

    //remove energy particles
    foreach(EnergyParticle* e, es) {
        _unit->getEnergyParticles().removeAll(e);
        _grid->removeEnergy(e->pos, e);
        delete e;
    }

    //remove cells
    QList< CellCluster* > allNewClusters;
    foreach(Cell* cell, cells) {

        //remove cell from cluster structure
        CellCluster* cluster = cell->getCluster();
        _grid->removeCellIfPresent(cluster->calcPosition(cell, _grid), cell);
        cluster->removeCell(cell, false);
        delete cell;

        //calculate new cluster structure
        QList< CellCluster* > newClusters;
        newClusters = cluster->decompose();
        _unit->getClusters() << newClusters;
        _unit->getClusters().removeAll(cluster);
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

void SimulationController::delExtendedSelection (QList< CellCluster* > clusters, QList< EnergyParticle* > es)
{
    _grid->lockData();

    //remove cell clusters
    foreach(CellCluster* cluster, clusters) {
        cluster->clearCellsFromMap();
        _unit->getClusters().removeAll(cluster);
        delete cluster;
    }

    //remove energy particles
    foreach(EnergyParticle* e, es) {
        _unit->getEnergyParticles().removeAll(e);
        _grid->removeEnergy(e->pos, e);
        delete e;
    }

    //clean up metadata
//    QSet< quint64 > ids = _grid->getAllCellIds();
//    _meta->cleanUp(ids);

    _grid->unlockData();
}

void SimulationController::rotateExtendedSelection (qreal angle, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _grid->lockData();

    //1. step: rotate each cluster around own center
    foreach(CellCluster* cluster, clusters) {
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
    foreach(CellCluster* cluster, clusters) {
        cluster->setPosition(transform.map(cluster->getPosition()));
    }
    foreach(EnergyParticle* e, es) {
        e->pos = transform.map(e->pos);
    }
    _grid->unlockData();
}

void SimulationController::setVelocityXExtendedSelection (qreal velX, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _grid->lockData();
    foreach(CellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVel();
        vel.setX(velX);
        cluster->setVel(vel);
    }
    foreach(EnergyParticle* e, es) {
        e->vel.setX(velX);
    }
    _grid->unlockData();
}

void SimulationController::setVelocityYExtendedSelection (qreal velY, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _grid->lockData();
    foreach(CellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVel();
        vel.setY(velY);
        cluster->setVel(vel);
    }
    foreach(EnergyParticle* e, es) {
        e->vel.setY(velY);
    }
    _grid->unlockData();
}

void SimulationController::setAngularVelocityExtendedSelection (qreal angVel, const QList< CellCluster* >& clusters)
{
    _grid->lockData();
    foreach(CellCluster* cluster, clusters) {
        cluster->setAngularVel(angVel);
    }
    _grid->unlockData();
}


QVector3D SimulationController::getCenterPosExtendedSelection (const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;

    _grid->lockData();
    foreach(CellCluster* cluster, clusters) {
        center += cluster->getPosition()*cluster->getMass();
/*        foreach(Cell* cell, cluster->getCellsRef())
            center += cell->calcPosition();*/
        numCells += cluster->getCellsRef().size();
    }
    foreach(EnergyParticle* e, es) {
        center += e->pos;
    }
    _grid->unlockData();

    center = center / (qreal)(numCells+es.size());
    return center;
}

void SimulationController::drawToMapExtendedSelection (const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _grid->lockData();
    foreach(CellCluster* cluster, clusters) {
        cluster->drawCellsToMap();
    }
    foreach(EnergyParticle* e, es) {
        _grid->setEnergy(e->pos, e);
    }
    _grid->unlockData();
}

void SimulationController::newCell (QVector3D pos)
{
    //create cluster with single cell
    _grid->lockData();
    FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    Cell* cell = facade->buildFeaturedCell(simulationParameters.NEW_CELL_ENERGY, CellFunctionType::COMPUTER
        , _grid, simulationParameters.NEW_CELL_MAX_CONNECTION, simulationParameters.NEW_CELL_TOKEN_ACCESS_NUMBER);
    cell->setTokenAccessNumber(_newCellTokenAccessNumber++);
    QList< Cell* > cells;
    cells << cell;
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, pos, 0, QVector3D(), _grid);
    _grid->setCell(pos, cell);
    _unit->getClusters() << cluster;
    _grid->unlockData();

    emit cellCreated(cell);
}

void SimulationController::newEnergyParticle (QVector3D pos)
{
    //create energy particle
    _grid->lockData();
    EnergyParticle* energy = new EnergyParticle(simulationParameters.CRIT_CELL_TRANSFORM_ENERGY/2, pos, QVector3D(), _grid);
    _grid->setEnergy(pos, energy);
    _unit->getEnergyParticles() << energy;
    _grid->unlockData();

    emit energyParticleCreated(energy);
}


void SimulationController::updateCell (QList< Cell* > cells, QList< CellTO > newCellsData, bool clusterDataChanged)
{
    //update purely cell data and no cluster data?
    if( !clusterDataChanged ) {
        _grid->lockData();

        QListIterator< Cell* > iCells(cells);
        QListIterator< CellTO > iNewCellsData(newCellsData);
        QSet< CellCluster* > sumNewClusters;
        FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
        while (iCells.hasNext()) {

            Cell* cell = iCells.next();
            CellTO newCellData = iNewCellsData.next();

            //update cell properties
            cell->getCluster()->updateTransformationMatrix();
            cell->setAbsPositionAndUpdateMap(newCellData.cellPos);
            cell->setEnergy(newCellData.cellEnergy);
            cell->delAllConnection();
            if( newCellData.cellMaxCon > simulationParameters.MAX_CELL_CONNECTIONS )
                newCellData.cellMaxCon = simulationParameters.MAX_CELL_CONNECTIONS;
            cell->setMaxConnections(newCellData.cellMaxCon);
            cell->setTokenBlocked(!newCellData.cellAllowToken);
            cell->setTokenAccessNumber(newCellData.cellTokenAccessNum);
            facade->changeFeaturesOfCell(cell, newCellData.cellFunctionType, _grid);
//            cell->setCellFunction(CellFunctionFactory::build(newCellData.cellFunctionName, false, _grid));

            //update cell computer
            CellFunctionComputer* computer = cell->getFeatures()->findObject<CellFunctionComputer>();
            if( computer ) {
                for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i ) {
                    computer->getMemoryReference()[i] = newCellData.computerMemory[i];
                }
                CellFunctionComputer::CompilationState state
                    = computer->injectAndCompileInstructionCode(newCellData.computerCode);
                if( !state.compilationOk )
                    emit computerCompilationReturn(true, state.errorAtLine);
                else
                    emit computerCompilationReturn(false, 0);
            }


            //update token
            cell->delAllTokens();
            int numToken = newCellData.tokenEnergies.size();
            for( int i = 0; i < numToken; ++i )
                cell->addToken(new Token(newCellData.tokenEnergies[i], newCellData.tokenData[i]), Cell::ACTIVATE_TOKEN::NOW, Cell::UPDATE_TOKEN_ACCESS_NUMBER::NO);

            //searching for nearby clusters
            QVector3D pos = cell->calcPosition();
            QSet< CellCluster* > clusters(_grid->getNearbyClusters(pos, qFloor(simulationParameters.CRIT_CELL_DIST_MAX+1.0)));
//            if( !clusters.contains(cell->getCluster()) )
            clusters << cell->getCluster();

            //update cell velocities
/*            foreach(CellCluster* cluster, clusters) {
                cluster->updateCellVel();
            }
*/
            //searching for nearby cells
            QList< Cell* > neighborCells;
            QList< QVector3D > neighborCellsRelPos;
            foreach(CellCluster* cluster, clusters)
                foreach(Cell* otherCell, cluster->getCellsRef()) {
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
                Cell* minCell = neighborCells.at(minIndex);
                neighborCells.removeAt(minIndex);
                neighborCellsRelPos.removeAt(minIndex);

                //still possible to make connection between cell and minCell?
                if( cell->connectable(minCell)) {

                    //make new connection
                    cell->newConnection(minCell);
                }
            }

            //calculating new cell clusters
            QList< CellCluster* > newClusters;
            foreach(CellCluster* cluster, clusters){//_thread->getClusters()) {
                newClusters << cluster->decompose();
                _unit->getClusters().removeAll(cluster);
                sumNewClusters.remove(cluster);
                delete cluster;
            }
            _unit->getClusters() << newClusters;

    //        if( _grid->getCell(cell->calcPosition()) == 0 )
    //            _grid->setCell(cell->calcPosition(), cell);
            sumNewClusters.unite(newClusters.toSet());
        }

        //update cell velocities
        QList< CellCluster* > changedClusters = sumNewClusters.toList();
        foreach(CellCluster* cluster, changedClusters)
            cluster->updateCellVel();

        _grid->unlockData();

        //inform other instances about reclustering
        emit reclustered(changedClusters);
    }
    //update only cluster data
    else {
        _grid->lockData();

        QListIterator< Cell* > iCells(cells);
        QListIterator< CellTO > iNewCellsData(newCellsData);
        QList< CellCluster* > sumNewClusters;
        while (iCells.hasNext()) {
            Cell* cell = iCells.next();
            CellTO newCellData = iNewCellsData.next();
            CellCluster* cluster = cell->getCluster();
            cluster->clearCellsFromMap();
            cluster->setPosition(newCellData.clusterPos);
            cluster->setAngle(newCellData.clusterAngle);
            cluster->setVel(newCellData.clusterVel);
            cluster->setAngularVel(newCellData.clusterAngVel);
            cluster->updateTransformationMatrix();
            cluster->drawCellsToMap();
            cluster->updateCellVel();
/*            foreach (Cell* otherCell, cluster->getCellsRef() ) {
                otherCell->setVel(QVector3D());
            }*/
            sumNewClusters<< cluster;
        }
        _grid->unlockData();
        emit reclustered(sumNewClusters);
    }
}

void SimulationController::setRun (bool run)
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
void SimulationController::forceFps (int fps)
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

void SimulationController::requestNextTimestep ()
{
    emit calcNextTimestep();
}

void SimulationController::updateUniverse ()
{
    emit universeUpdated(_grid, true);
}

void SimulationController::forceFpsTimerSlot ()
{
    if( _run ) {
        if( !_calculating ) {
            _calculating = true;
            emit calcNextTimestep();
        }
    }
}

void SimulationController::nextTimestepCalculated ()
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

