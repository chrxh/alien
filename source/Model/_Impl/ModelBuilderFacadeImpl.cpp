#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Particle.h"
#include "Model/Entities/Token.h"
#include "Model/Entities/EntityFactory.h"
#include "Model/Features/CellFunction.h"
#include "Model/Features/CellComputer.h"
#include "Model/Features/EnergyGuidance.h"
#include "Model/Features/CellFeatureFactory.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/SpaceMetric.h"
#include "Model/Context/ContextFactory.h"
#include "Model/Context/MapCompartment.h"
#include "Model/Context/UnitThreadController.h"
#include "Model/Context/UnitGrid.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/Unit.h"
#include "Model/Context/UnitContext.h"
#include "Model/AccessPorts/AccessPortFactory.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/Settings.h"
#include "Model/_Impl/SimulationControllerImpl.h"

#include "ModelBuilderFacadeImpl.h"
#include "CellConnectorImpl.h"

namespace
{
	const int ARRAY_SIZE_FOR_RANDOM_NUMBERS = 234327;
}

SimulationController* ModelBuilderFacadeImpl::buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
	, SymbolTable* symbolTable, SimulationParameters* parameters) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	SimulationContext* context = contextFactory->buildSimulationContext();

	auto threads = contextFactory->buildSimulationThreads();
	auto grid = contextFactory->buildSimulationGrid();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = contextFactory->buildSpaceMetric();
	metric->init(universeSize);
	threads->init(maxRunngingThreads);
	grid->init(gridSize, metric);
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, 0);
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

	auto controller = new SimulationControllerImpl();
	controller->init(static_cast<SimulationContext*>(context));

	return controller;
}

SimulationAccess * ModelBuilderFacadeImpl::buildSimulationAccess(SimulationContextApi * contextApi) const
{
	AccessPortFactory* factory = ServiceLocator::getInstance().getService<AccessPortFactory>();
	auto access = factory->buildSimulationAccess();
	access->init(contextApi);
	return access;
}

CellConnector * ModelBuilderFacadeImpl::buildCellConnector(SimulationContextApi* contextApi) const
{
	auto connector = new CellConnectorImpl();
	auto context = static_cast<SimulationContext*>(contextApi);
	connector->init(context->getSpaceMetric(), context->getSimulationParameters(), context->getNumberGenerator());
	return connector;
}

Unit * ModelBuilderFacadeImpl::buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto grid = context->getUnitGrid();
	auto threads = context->getUnitThreadController();

	auto unit = contextFactory->buildSimulationUnit();		//unit has no parent due to an QObject::moveToThread call later
	auto unitContext = contextFactory->buildSimulationUnitContext();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = static_cast<SpaceMetric*>(context->getSpaceMetric())->clone();
	auto compartment = contextFactory->buildMapCompartment();
	auto cellMap = contextFactory->buildCellMap();
	auto energyMap = contextFactory->buildEnergyParticleMap();
	auto symbolTable = context->getSymbolTable()->clone();
	auto parameters = context->getSimulationParameters()->clone();
	uint16_t threadId = gridPos.x + gridPos.y * grid->getSize().x + 1;
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, threadId);
	compartment->init(grid->calcCompartmentRect(gridPos));
	cellMap->init(metric, compartment);
	energyMap->init(metric, compartment);
	unitContext->init(numberGen, metric, cellMap, energyMap, compartment, symbolTable, parameters);
	unit->init(unitContext);

	return unit;
}

SymbolTable * ModelBuilderFacadeImpl::buildDefaultSymbolTable() const
{
	return ModelSettings::loadDefaultSymbolTable();
}

SimulationParameters * ModelBuilderFacadeImpl::buildDefaultSimulationParameters() const
{
	return ModelSettings::loadDefaultSimulationParameters();
}
