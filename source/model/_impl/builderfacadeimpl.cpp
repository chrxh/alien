#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"
#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EnergyParticle.h"
#include "model/entities/Token.h"
#include "model/entities/EntityFactory.h"
#include "model/features/CellFunction.h"
#include "model/features/CellFunctionComputer.h"
#include "model/features/EnergyGuidance.h"
#include "model/features/CellFeatureFactory.h"
#include "model/context/SimulationParameters.h"
#include "model/context/CellMap.h"
#include "model/context/EnergyParticleMap.h"
#include "model/context/SpaceMetric.h"
#include "model/context/ContextFactory.h"
#include "model/context/MapCompartment.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/SimulationContext.h"
#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/AccessPorts/AccessPortFactory.h"
#include "model/AccessPorts/SimulationAccess.h"
#include "model/metadata/SymbolTable.h"
#include "model/Settings.h"
#include "model/_impl/SimulationControllerImpl.h"

#include "BuilderFacadeImpl.h"

namespace {
	BuilderFacadeImpl instance;
}

BuilderFacadeImpl::BuilderFacadeImpl ()
{
    ServiceLocator::getInstance().registerService<BuilderFacade>(this);
}

SimulationAccess * BuilderFacadeImpl::buildSimulationAccess(SimulationContextApi * context) const
{
	AccessPortFactory* factory = ServiceLocator::getInstance().getService<AccessPortFactory>();
	auto access = factory->buildSimulationAccess();
	access->init(context);
	return access;
}

SimulationController * BuilderFacadeImpl::buildSimulationController(SimulationContextApi * context) const
{
	auto controller = new SimulationControllerImpl();
	controller->init(static_cast<SimulationContext*>(context));
	return controller;
}

SimulationContextApi* BuilderFacadeImpl::buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric, SymbolTable* symbolTable
	, SimulationParameters* parameters) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	SimulationContext* context = contextFactory->buildSimulationContext();

	auto threads = contextFactory->buildSimulationThreads();
	auto grid = contextFactory->buildSimulationGrid();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	threads->init(maxRunngingThreads);
	grid->init(gridSize, metric);
	context->init(numberGen, metric, grid, threads, symbolTable, parameters);

	for (int x = 0; x < gridSize.x; ++x) {
		for (int y = 0; y < gridSize.y; ++y) {
			auto unit = buildSimulationUnit({ x,y }, context);
			grid->registerUnit({ x,y }, unit);
			threads->registerUnit(unit);
		}
	}

	for (int x = 0; x < gridSize.x; ++x) {
		for (int y = 0; y < gridSize.y; ++y) {
			auto unitContext = grid->getUnitOfGridPos({ x, y })->getContext();
			auto compartment = unitContext->getMapCompartment();
			auto getContextFromDelta = [&](IntVector2D const& delta) {
				return grid->getUnitOfGridPos({ (x + delta.x + gridSize.x) % gridSize.x, (y + delta.y + gridSize.y) % gridSize.y })->getContext();
			};
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::UpperLeft, getContextFromDelta({ -1, -1 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::Upper, getContextFromDelta({ 0, -1 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::UpperRight, getContextFromDelta({ +1, -1 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::Left, getContextFromDelta({ -1, 0 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::Right, getContextFromDelta({ +1, 0 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::LowerLeft, getContextFromDelta({ -1, +1 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::Lower, getContextFromDelta({ 0, +1 }));
			compartment->registerNeighborContext(MapCompartment::RelativeLocation::LowerRight, getContextFromDelta({ +1, +1 }));
		}
	}

	return context;
}

namespace
{
	const int ARRAY_SIZE_FOR_RANDOM_NUMBERS = 234327;
}

Unit * BuilderFacadeImpl::buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto grid = context->getUnitGrid();
	auto threads = context->getUnitThreadController();

	auto unit = contextFactory->buildSimulationUnit();		//unit has no parent due to an QObject::moveToThread call later
	auto unitContext = contextFactory->buildSimulationUnitContext();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = context->getSpaceMetric()->clone();
	auto compartment = contextFactory->buildMapCompartment();
	auto cellMap = contextFactory->buildCellMap();
	auto energyMap = contextFactory->buildEnergyParticleMap();
	auto symbolTable = context->getSymbolTable()->clone();
	auto parameters = context->getSimulationParameters()->clone();
	uint16_t threadId = gridPos.x + gridPos.y * grid->getSize().x;
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, threadId);
	compartment->init(grid->calcCompartmentRect(gridPos));
	cellMap->init(metric, compartment);
	energyMap->init(metric, compartment);
	unitContext->init(numberGen, metric, cellMap, energyMap, compartment, symbolTable, parameters);
	unit->init(unitContext);

	return unit;
}

SpaceMetric * BuilderFacadeImpl::buildSpaceMetric(IntVector2D universeSize) const
{
	ContextFactory* factory = ServiceLocator::getInstance().getService<ContextFactory>();
	auto metric = factory->buildSpaceMetric();
	metric->init(universeSize);
	return metric;
}

SymbolTable * BuilderFacadeImpl::buildDefaultSymbolTable() const
{
	return ModelSettings::loadDefaultSymbolTable();
}

SimulationParameters * BuilderFacadeImpl::buildDefaultSimulationParameters() const
{
	return ModelSettings::loadDefaultSimulationParameters();
}
