#include "global/servicelocator.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/entities/token.h"
#include "model/entities/entityfactory.h"
#include "model/features/cellfunction.h"
#include "model/features/cellfunctioncomputer.h"
#include "model/features/energyguidance.h"
#include "model/features/cellfeaturefactory.h"
#include "model/simulationparameters.h"
#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/topology.h"
#include "model/config.h"
#include "model/_impl/simulationcontextimpl.h"

#include "alienfacadeimpl.h"

namespace {
	AlienFacadeImpl factoryFacadeImpl;
}

AlienFacadeImpl::AlienFacadeImpl ()
{
    ServiceLocator::getInstance().registerService<AlienFacade>(this);
}

SimulationContext* AlienFacadeImpl::buildSimulationContext() const
{
	SimulationContext* context = new SimulationContextImpl();
	Metadata::loadDefaultSymbolTable(context->getSymbolTable());
	Metadata::loadDefaultSimulationParameters(context->getSimulationParameters());
	return context;
}

CellCluster* AlienFacadeImpl::buildCellCluster (SimulationContext* context) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    return entityFactory->buildCellCluster(context);
}

CellCluster* AlienFacadeImpl::buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
    , QVector3D vel, SimulationContext* context) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    return entityFactory->buildCellCluster(cells, angle, pos, angularVel, vel, context);
}


Cell* AlienFacadeImpl::buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data
    , SimulationContext* context, int maxConnections, int tokenAccessNumber, QVector3D relPos) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildCell(energy, context, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, data, context);
    decoratorFactory->addEnergyGuidance(cell, context);
    return cell;
}

Cell* AlienFacadeImpl::buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, SimulationContext* context
    , int maxConnections, int tokenAccessNumber, QVector3D relPos) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildCell(energy, context, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, context);
    decoratorFactory->addEnergyGuidance(cell, context);
    return cell;
}

Cell* AlienFacadeImpl::buildFeaturedCellWithRandomData (qreal energy, SimulationContext* context) const
{
	SimulationParameters* parameters = context->getSimulationParameters();
    int randomMaxConnections = qrand() % (parameters->MAX_CELL_CONNECTIONS+1);
    int randomTokenAccessNumber = qrand() % parameters->MAX_TOKEN_ACCESS_NUMBERS;
    QByteArray randomData(256, 0);
	for (int i = 0; i < 256; ++i)
		randomData[i] = qrand() % 256;
    Enums::CellFunction::Type randomCellFunction = static_cast<Enums::CellFunction::Type>(qrand() % Enums::CellFunction::_COUNTER);
    return buildFeaturedCell(energy, randomCellFunction, randomData, context, randomMaxConnections, randomTokenAccessNumber, QVector3D());
}

CellTO AlienFacadeImpl::buildFeaturedCellTO (Cell* cell) const
{
    CellTO to;

    //copy cell properties
    CellCluster* cluster = cell->getCluster();
    to.numCells = cluster->getMass();
    to.clusterPos = cluster->getPosition();
    to.clusterVel = cluster->getVel();
    to.clusterAngle = cluster->getAngle();
    to.clusterAngVel = cluster->getAngularVel();
    to.cellPos = cell->calcPosition();
    to.cellEnergy = cell->getEnergy();
    to.cellNumCon = cell->getNumConnections();
    to.cellMaxCon = cell->getMaxConnections();
    to.cellAllowToken = !cell->isTokenBlocked();
    to.cellTokenAccessNum = cell->getBranchNumber();
    CellFunction* cellFunction = cell->getFeatures()->findObject<CellFunction>();
    to.cellFunctionType = cellFunction->getType();

    //copy computer data
    CellFunctionComputer* computer = cellFunction->findObject<CellFunctionComputer>();
    if( computer ) {
        QByteArray d = computer->getMemoryReference();
		to.computerMemory = d;
        to.computerCode = computer->decompileInstructionCode();
    }

    //copy token data
    for(int i = 0; i < cell->getNumToken(); ++i) {
        Token* token = cell->getToken(i);
        to.tokenEnergies << token->getEnergy();
        to.tokenData << token->getMemoryRef();
    }
    return to;
}

void AlienFacadeImpl::changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, SimulationContext* context) const
{
    cell->removeFeatures();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    decoratorFactory->addCellFunction(cell, type, context);
    decoratorFactory->addEnergyGuidance(cell, context);
}

