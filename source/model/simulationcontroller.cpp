#include <QTimer>
#include <QtCore/qmath.h>
#include <QVector2D>
#include <QMatrix4x4>
#include <set>
#include <vector>

#include "global/global.h"
#include "global/servicelocator.h"
#include "model/features/cellfunctioncomputer.h"
#include "model/entities/entityfactory.h"
#include "model/entities/token.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/metadata/symboltable.h"
#include "model/physics/physics.h"
#include "model/config.h"
#include "simulationcontext.h"
#include "simulationparameters.h"
#include "simulationunit.h"
#include "energyparticlemap.h"
#include "cellmap.h"
#include "topology.h"
#include "serializationfacade.h"
#include "alienfacade.h"

#include "simulationcontroller.h"

SimulationController::SimulationController(Threading threading, QObject* parent)
    : QObject(parent)
	, _threading(threading)
{
	AlienFacade* factory = ServiceLocator::getInstance().getService<AlienFacade>();

    _forceFpsTimer = new QTimer(this);
	_oneSecondTimer = new QTimer(this);
	_context = factory->buildSimulationContext();
    _unit = new SimulationUnit(_context);

	connect(_oneSecondTimer, SIGNAL(timeout()), this, SLOT(oneSecondTimerSlot()));
    connect(_forceFpsTimer, SIGNAL(timeout()), this, SLOT(forceFpsTimerSlot()));
	connect(this, SIGNAL(initUnit(uint)), _unit, SLOT(init(uint)));

	_oneSecondTimer->start(1000);

    if( threading == Threading::EXTRA_THREAD ) {
        connect(this, SIGNAL(calcNextTimestep()), _unit, SLOT(calcNextTimestep()));
        connect(_unit, SIGNAL(nextTimestepCalculated()), this, SLOT(nextTimestepCalculated()));

		startThread();
    }
    if( threading == Threading::NO_EXTRA_THREAD ) {
        connect(this, SIGNAL(calcNextTimestep()), _unit, SLOT(calcNextTimestep()), Qt::DirectConnection);
        _unit->setParent(this);
    }

	emit initUnit(0);
}

SimulationController::~SimulationController ()
{
	terminateThread();
	delete _context;
}

void SimulationController::startThread()
{
	_thread = new QThread(this);
	connect(_thread, &QThread::finished, _unit, &QObject::deleteLater);
	_unit->moveToThread(_thread);
	_thread->start();
}

void SimulationController::terminateThread()
{
	if (!_thread) {
		return;
	}
	_thread->quit();
	if (!_thread->wait(2000)) {
		_thread->terminate();
		_thread->wait();
	}
}

QMap< QString, qreal > SimulationController::getMonitorData ()
{
    QMap< QString, qreal > data;
    _context->lock();
    int cells(0);
    int particles(0);
    int token(0);
    qreal internalEnergy(_unit->calcInternalEnergy());
    foreach( CellCluster* cluster, _context->getClustersRef() ) {
        cells += cluster->getCellsRef().size();
        foreach( Cell* cell, cluster->getCellsRef() ) {
            token += cell->getNumToken();
        }
    }
    particles = _context->getEnergyParticlesRef().size();
	_context->unlock();
	data["cells"] = cells;
    data["clusters"] = _context->getClustersRef().size();
    data["energyParticles"] = particles;
    data["token"] = token;
    data["internalEnergy"] = internalEnergy;
    data["transEnergy"] = _unit->calcTransEnergy() / _context->getSimulationParameters()->cellMass_Reciprocal;
    data["rotEnergy"] = _unit->calcRotEnergy()/ _context->getSimulationParameters()->cellMass_Reciprocal;
    return data;
}

SimulationContext* SimulationController::getSimulationContext()
{
    return _context;
}

void SimulationController::newUniverse (IntVector2D size, SymbolTable const& symbolTable)
{
	_context->lock();

	_frame = 0;

	_context->init(size);
	_context->getSymbolTable()->setTable(symbolTable);
	_context->getClustersRef().clear();
	_context->getEnergyParticlesRef().clear();

	_context->unlock();
}

void SimulationController::saveUniverse (QDataStream& stream)
{
    emit initUnit(_frame);	//reset random seed for simulation thread to be deterministic

	_context->lock();
	stream << _frame;
	SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();

	std::set<quint64> ids = _context->getAllCellIds();
	facade->serializeSimulationContext(_context, stream);
    _context->unlock();
}

void SimulationController::loadUniverse(QDataStream& stream)
{
	_context->lock();
	stream >> _frame;
	SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
	facade->deserializeSimulationContext(_context, stream);
	_unit->setContext(_context);
	_context->unlock();

	emit initUnit(_frame);	//reset random seed for simulation thread to be deterministic

/*    simulationParameters.readData(stream);
    MetadataManager::getGlobalInstance().readMetadataUniverse(stream, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readSymbolTable(stream);
	*/
}

IntVector2D SimulationController::getUniverseSize()
{
    _context->lock();
	IntVector2D size = _context->getTopology()->getSize();
    _context->unlock();
    return size;
}

void SimulationController::addBlockStructure (QVector3D center, int numCellX, int numCellY, QVector3D dist, qreal energy)
{
    //create cell grid
    AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
	std::vector<std::vector<Cell*>> cellGrid;
	cellGrid.resize(numCellX);
	for (int x = 0; x < numCellX; ++x) {
		cellGrid[x].resize(numCellY);
	}
    for(int i = 0; i < numCellX; ++i )
        for(int j = 0; j < numCellY; ++j ) {
            qreal x = - ((qreal)numCellX-1.0)*dist.x()/2.0 + (qreal)i*dist.x();
            qreal y = - ((qreal)numCellY-1.0)*dist.y()/2.0 + (qreal)j*dist.y();
            int maxCon = 4;
            if( (i == 0) || (i == (numCellX-1)) || (j == 0) || (j == (numCellY-1)) )
                maxCon = 3;
            if( ((i == 0) || (i == (numCellX-1))) && ((j == 0) || (j == (numCellY-1))) )
                maxCon = 2;
            Cell* cell = facade->buildFeaturedCell(energy, Enums::CellFunction::COMPUTER, _context, maxCon, 0, QVector3D(x, y, 0.0));

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
    _context->lock();
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _context);
    cluster->drawCellsToMap();
    _context->getClustersRef() << cluster;
    _context->unlock();
    QList< CellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void SimulationController::addHexagonStructure (QVector3D center, int numLayers, qreal dist, qreal energy)
{
    //create hexagon cell structure
    AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
	std::vector<std::vector<Cell*>> cellGrid;
	int size = 2 * numLayers - 1;
	cellGrid.resize(size);
	for (int x = 0; x < size; ++x) {
		cellGrid[x].resize(size);
	}
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
            cellGrid[numLayers-1+i][numLayers-1-j] = facade->buildFeaturedCell(energy, Enums::CellFunction::COMPUTER, _context, maxCon, 0, QVector3D(i*dist+j*dist/2.0, -j*incY, 0.0));
            cells << cellGrid[numLayers-1+i][numLayers-1-j];
            if( numLayers-1+i > 0 )
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i-1][numLayers-1-j]);
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i][numLayers-1-j+1]);
                cellGrid[numLayers-1+i][numLayers-1-j]->newConnection(cellGrid[numLayers-1+i+1][numLayers-1-j+1]);
            }

            //create cell: under layer (except for 0-layer)
            if( j > 0 ) {
                cellGrid[numLayers-1+i][numLayers-1+j] = facade->buildFeaturedCell(energy, Enums::CellFunction::COMPUTER, _context, maxCon, 0, QVector3D(i*dist+j*dist/2.0, +j*incY, 0.0));
                cells << cellGrid[numLayers-1+i][numLayers-1+j];
                if( numLayers-1+i > 0 )
                    cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i-1][numLayers-1+j]);
                cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i][numLayers-1+j-1]);
                cellGrid[numLayers-1+i][numLayers-1+j]->newConnection(cellGrid[numLayers-1+i+1][numLayers-1+j-1]);
            }
        }
    }

    //create cluster
    _context->lock();
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, center, 0.0, QVector3D(), _context);
    cluster->drawCellsToMap();
    _context->getClustersRef() << cluster;
    _context->unlock();
    QList< CellCluster* > newCluster;
    newCluster << cluster;
    emit reclustered(newCluster);
}

void SimulationController::addRandomEnergy (qreal energy, qreal maxEnergyPerParticle)
{
    _context->lock();

    while( energy > 0.0 ) {
        qreal rEnergy = GlobalFunctions::random(0.01, maxEnergyPerParticle);
        if( rEnergy > energy )
            rEnergy = energy;
        qreal posX = GlobalFunctions::random(0.0, _context->getTopology()->getSize().x);
        qreal posY = GlobalFunctions::random(0.0, _context->getTopology()->getSize().x);
        qreal velX = GlobalFunctions::random(-1.0, 1.0);
        qreal velY = GlobalFunctions::random(-1.0, 1.0);
		auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
        EnergyParticle* e = factory->buildEnergyParticle(rEnergy, QVector3D(posX, posY, 0.0), QVector3D(velX, velY, 0.0), _context);
        _context->getEnergyParticlesRef() << e;
        _context->getEnergyParticleMap()->setParticle(e->getPosition(), e);
        energy -= rEnergy;
    }
    _context->unlock();
}

void SimulationController::saveCell (QDataStream& stream, Cell* cell, quint64& clusterId, quint64& cellId)
{
    _context->lock();

    //serialize cell data
    SerializationFacade *facade = ServiceLocator::getInstance().getService<SerializationFacade>();
    facade->serializeFeaturedCell(cell, stream);
    stream << clusterId;

    //return ids
    CellCluster* cluster = cell->getCluster();
    clusterId = cluster->getId();
    cellId = cell->getId();

    _context->unlock();
}

void SimulationController::saveExtendedSelection (QDataStream& stream, const QList< CellCluster* >& clusters
	, const QList< EnergyParticle* >& es, QList< quint64 >& clusterIds, QList< quint64 >& cellIds)
{
	_context->lock();

	SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();

	//serialize num clusters
    quint32 numClusters = clusters.size();
    stream << numClusters;

    //serialize cluster data
    foreach(CellCluster* cluster, clusters) {
		facade->serializeCellCluster(cluster, stream);
        clusterIds << cluster->getId();
        foreach(Cell* cell, cluster->getCellsRef())
            cellIds << cell->getId();
    }

    //serialize num energy particles
    quint32 numEs = es.size();
    stream << numEs;

    //serialize energy particle data
    foreach(EnergyParticle* e, es) {
		facade->serializeEnergyParticle(e, stream);
    }

    _context->unlock();
}

void SimulationController::loadCell(QDataStream& stream, QVector3D pos, bool drawToMap /*= true*/)
{
    _context->lock();

    SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
	AlienFacade* factory = ServiceLocator::getInstance().getService<AlienFacade>();

	QList< Cell* > newCells;
	Cell* newCell = facade->deserializeFeaturedCell(stream, _context);
    newCells << newCell;
    newCells[0]->setRelPosition(QVector3D());
    CellCluster* newCluster = factory->buildCellCluster(newCells, 0, pos, 0, newCell->getVelocity(), _context);
    _context->getClustersRef() << newCluster;

    //read old cluster id
    quint64 oldClusterId = 0;
    stream >> oldClusterId;

//->	MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);

    //draw cluster
    if( drawToMap )
        newCluster->drawCellsToMap();

    _context->unlock();
}

int SimulationController::getFrame() const
{
	return _frame;
}

int SimulationController::getFps() const
{
	return _fps;
}


void SimulationController::loadExtendedSelection (QDataStream& stream, QVector3D pos, QList< CellCluster* >& newClusters
	, QList< EnergyParticle* >& newEnergyParticles, QMap< quint64, quint64 >& oldNewClusterIdMap
	, QMap< quint64, quint64 >& oldNewCellIdMap, bool drawToMap)
{
    _context->lock();

    //maps for associating new cells and energy particles
    QMap< quint64, Cell* > oldIdCellMap;

    //read num clusters
    quint32 numClusters;
    stream >> numClusters;

    //read cluster data
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;
    SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
    for(int i = 0; i < numClusters; ++i) {
        CellCluster* cluster = facade->deserializeCellCluster(stream, _context);
        newClusters << cluster;
        foreach(Cell* cell, cluster->getCellsRef())
            center += cell->calcPosition();
        numCells += cluster->getCellsRef().size();
    }
    _context->getClustersRef() << newClusters;

    //read num energy particles
    quint32 numEnergyParticles;
    stream >> numEnergyParticles;

    //read energy particle data
    for(int i = 0; i < numEnergyParticles; ++i) {
        EnergyParticle* e = facade->deserializeEnergyParticle(stream, _context);
        center += e->getPosition();
        newEnergyParticles << e;
    }
    _context->getEnergyParticlesRef() << newEnergyParticles;

    //set new center and draw cluster
    center = center / (qreal)(numCells+numEnergyParticles);
    foreach(CellCluster* cluster, newClusters) {
        cluster->setCenterPosition(cluster->getPosition()-center+pos);
        cluster->updateTransformationMatrix();
        if( drawToMap )
            cluster->drawCellsToMap();
    }
    foreach(EnergyParticle* e, newEnergyParticles) {
        e->setPosition(e->getPosition()-center+pos);
        if( drawToMap )
            _context->getEnergyParticleMap()->setParticle(e->getPosition(), e);
    }
    _context->unlock();
}

void SimulationController::delSelection (QList< Cell* > cells, QList< EnergyParticle* > es)
{
    _context->lock();

    //remove energy particles
    foreach(EnergyParticle* e, es) {
        _context->getEnergyParticlesRef().removeAll(e);
		_context->getEnergyParticleMap()->removeParticleIfPresent(e->getPosition(), e);
        delete e;
    }

    //remove cells
    QList< CellCluster* > allNewClusters;
    foreach(Cell* cell, cells) {

        //remove cell from cluster structure
        CellCluster* cluster = cell->getCluster();
        _context->getCellMap()->removeCellIfPresent(cluster->calcPosition(cell), cell);
        cluster->removeCell(cell, false);
        delete cell;

        //calculate new cluster structure
        QList< CellCluster* > newClusters;
        newClusters = cluster->decompose();
        _context->getClustersRef() << newClusters;
        _context->getClustersRef().removeAll(cluster);
        allNewClusters.removeAll(cluster);
        allNewClusters << newClusters;
        delete cluster;
    }

    //clean up metadata
//    QSet< quint64 > ids = _grid->getAllCellIds();
//    _meta->cleanUp(ids);

    _context->unlock();
//????
//    if( !allNewClusters.isEmpty() )
        emit reclustered(allNewClusters);
}

void SimulationController::delExtendedSelection (QList< CellCluster* > clusters, QList< EnergyParticle* > es)
{
    _context->lock();

    //remove cell clusters
    foreach(CellCluster* cluster, clusters) {
        cluster->clearCellsFromMap();
        _context->getClustersRef().removeAll(cluster);
        delete cluster;
    }

    //remove energy particles
    foreach(EnergyParticle* e, es) {
        _context->getEnergyParticlesRef().removeAll(e);
		_context->getEnergyParticleMap()->removeParticleIfPresent(e->getPosition(), e);
        delete e;
    }

    _context->unlock();
}

void SimulationController::rotateExtendedSelection (qreal angle, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _context->lock();

    //1. step: rotate each cluster around own center
    foreach(CellCluster* cluster, clusters) {
        cluster->setAngle(cluster->getAngle()+angle);
    }

    //2. step: rotate cluster around common center
    _context->unlock();
    QVector3D center = getCenterPosExtendedSelection(clusters, es);
    _context->lock();
    QMatrix4x4 transform;
    transform.setToIdentity();
    transform.translate(center);
    transform.rotate(angle, 0.0, 0.0, 1.0);
    transform.translate(-center);
    foreach(CellCluster* cluster, clusters) {
        cluster->setCenterPosition(transform.map(cluster->getPosition()));
    }
    foreach(EnergyParticle* e, es) {
        e->setPosition(transform.map(e->getPosition()));
    }
    _context->unlock();
}

void SimulationController::setVelocityXExtendedSelection (qreal velX, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _context->lock();
    foreach(CellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVelocity();
        vel.setX(velX);
        cluster->setVelocity(vel);
    }
    foreach(EnergyParticle* e, es) {
		auto vel = e->getVelocity();
		vel.setX(velX);
		e->setVelocity(vel);
    }
    _context->unlock();
}

void SimulationController::setVelocityYExtendedSelection (qreal velY, const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _context->lock();
    foreach(CellCluster* cluster, clusters) {
        QVector3D vel = cluster->getVelocity();
        vel.setY(velY);
        cluster->setVelocity(vel);
    }
    foreach(EnergyParticle* e, es) {
		auto vel = e->getVelocity();
		vel.setY(velY);
		e->setVelocity(vel);
    }
    _context->unlock();
}

void SimulationController::setAngularVelocityExtendedSelection (qreal angVel, const QList< CellCluster* >& clusters)
{
    _context->lock();
    foreach(CellCluster* cluster, clusters) {
        cluster->setAngularVel(angVel);
    }
    _context->unlock();
}


QVector3D SimulationController::getCenterPosExtendedSelection (const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    QVector3D center(0.0, 0.0, 0.0);
    quint32 numCells = 0;

    _context->lock();
    foreach(CellCluster* cluster, clusters) {
        center += cluster->getPosition()*cluster->getMass();
/*        foreach(Cell* cell, cluster->getCellsRef())
            center += cell->calcPosition();*/
        numCells += cluster->getCellsRef().size();
    }
    foreach(EnergyParticle* e, es) {
        center += e->getPosition();
    }
    _context->unlock();

    center = center / (qreal)(numCells+es.size());
    return center;
}

void SimulationController::drawToMapExtendedSelection (const QList< CellCluster* >& clusters, const QList< EnergyParticle* >& es)
{
    _context->lock();
    foreach(CellCluster* cluster, clusters) {
        cluster->drawCellsToMap();
    }
    foreach(EnergyParticle* e, es) {
		_context->getEnergyParticleMap()->setParticle(e->getPosition(), e);
    }
    _context->unlock();
}

void SimulationController::newCell (QVector3D pos)
{
    //create cluster with single cell
    _context->lock();
    AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
	SimulationParameters* paramters = _context->getSimulationParameters();
    Cell* cell = facade->buildFeaturedCell(paramters->cellCreationEnergy, Enums::CellFunction::COMPUTER
        , _context, paramters->NEW_CELL_MAX_CONNECTION, paramters->NEW_CELL_TOKEN_ACCESS_NUMBER);
    cell->setBranchNumber(_newCellTokenAccessNumber++);
    QList< Cell* > cells;
    cells << cell;
    CellCluster* cluster = facade->buildCellCluster(cells, 0.0, pos, 0, QVector3D(), _context);
	_context->getCellMap()->setCell(pos, cell);
    _context->getClustersRef() << cluster;
    _context->unlock();

    emit cellCreated(cell);
}

void SimulationController::newEnergyParticle (QVector3D pos)
{
    //create energy particle
    _context->lock();
	auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
    EnergyParticle* energy = factory->buildEnergyParticle(_context->getSimulationParameters()->cellMinEnergy/2, pos, QVector3D(), _context);
	_context->getEnergyParticleMap()->setParticle(pos, energy);
    _context->getEnergyParticlesRef() << energy;
    _context->unlock();

    emit energyParticleCreated(energy);
}


void SimulationController::updateCell (QList< Cell* > cells, QList< CellTO > newCellsData, bool clusterDataChanged)
{
    //update purely cell data and no cluster data?
    if( !clusterDataChanged ) {
        _context->lock();

        QListIterator< Cell* > iCells(cells);
        QListIterator< CellTO > iNewCellsData(newCellsData);
        QSet< CellCluster* > sumNewClusters;
        AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
		EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
		SimulationParameters* parameters = _context->getSimulationParameters();
        while (iCells.hasNext()) {

            Cell* cell = iCells.next();
            CellTO newCellData = iNewCellsData.next();

            //update cell properties
            cell->getCluster()->updateTransformationMatrix();
            cell->setAbsPositionAndUpdateMap(newCellData.cellPos);
            cell->setEnergy(newCellData.cellEnergy);
            cell->delAllConnection();
            if( newCellData.cellMaxCon > parameters->cellMaxBonds )
                newCellData.cellMaxCon = parameters->cellMaxBonds;
            cell->setMaxConnections(newCellData.cellMaxCon);
            cell->setTokenBlocked(!newCellData.cellAllowToken);
            cell->setBranchNumber(newCellData.cellTokenAccessNum);
            facade->changeFeaturesOfCell(cell, newCellData.cellFunctionType, _context);
//            cell->setCellFunction(CellFunctionFactory::build(newCellData.cellFunctionName, false, _grid));

            //update cell computer
            CellFunctionComputer* computer = cell->getFeatures()->findObject<CellFunctionComputer>();
            if( computer ) {
                for( int i = 0; i < parameters->cellFunctionComputerCellMemorySize; ++i ) {
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
			for (int i = 0; i < numToken; ++i) {
				auto token = entityFactory->buildToken(_context, newCellData.tokenEnergies[i], newCellData.tokenData[i]);
				cell->addToken(token, Cell::ActivateToken::NOW, Cell::UpdateTokenAccessNumber::NO);
			}

            //searching for nearby clusters
            QVector3D pos = cell->calcPosition();
            CellClusterSet clusters = _context->getCellMap()->getNearbyClusters(pos, qFloor(parameters->cellMaxDistance+1.0));
//            if( !clusters.contains(cell->getCluster()) )
            clusters.insert(cell->getCluster());

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
                    _context->getTopology()->correctDisplacement(displacement);
                    qreal dist = displacement.length();
                    if (cell != otherCell && dist < parameters->cellMaxDistance) {

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
                _context->getClustersRef().removeAll(cluster);
                sumNewClusters.remove(cluster);
                delete cluster;
            }
            _context->getClustersRef() << newClusters;

    //        if( _grid->getCell(cell->calcPosition()) == 0 )
    //            _grid->setCell(cell->calcPosition(), cell);
            sumNewClusters.unite(newClusters.toSet());
        }

        //update cell velocities
        QList< CellCluster* > changedClusters = sumNewClusters.toList();
        foreach(CellCluster* cluster, changedClusters)
            cluster->updateCellVel();

        _context->unlock();

        //inform other instances about reclustering
        emit reclustered(changedClusters);
    }
    //update only cluster data
    else {
        _context->lock();

        QListIterator< Cell* > iCells(cells);
        QListIterator< CellTO > iNewCellsData(newCellsData);
        QList< CellCluster* > sumNewClusters;
        while (iCells.hasNext()) {
            Cell* cell = iCells.next();
            CellTO newCellData = iNewCellsData.next();
            CellCluster* cluster = cell->getCluster();
            cluster->clearCellsFromMap();
            cluster->setCenterPosition(newCellData.clusterPos);
            cluster->setAngle(newCellData.clusterAngle);
            cluster->setVelocity(newCellData.clusterVel);
            cluster->setAngularVel(newCellData.clusterAngVel);
            cluster->updateTransformationMatrix();
            cluster->drawCellsToMap();
            cluster->updateCellVel();
/*            foreach (Cell* otherCell, cluster->getCellsRef() ) {
                otherCell->setVel(QVector3D());
            }*/
            sumNewClusters<< cluster;
        }
        _context->unlock();
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
    _forceFps = fps;
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
    emit universeUpdated(_context, true);
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

void SimulationController::oneSecondTimerSlot()
{
	_fps = _frame - _frameFromLastSecond;
	_frameFromLastSecond = _frame;
}

void SimulationController::nextTimestepCalculated ()
{
    _frame++;
    _calculating = false;
    emit universeUpdated(_context, false);

    //fps forcing must be deactivate in order to continue with next the frame
    if( _run ) {
        if( _forceFps == 0 )
            emit calcNextTimestep();
    }
}

