#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"

#include "Model/Api/SimulationParameters.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/Settings.h"
#include "Model/Local/Cell.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Particle.h"
#include "Model/Local/Token.h"
#include "Model/Local/EntityFactory.h"
#include "Model/Local/CellFunction.h"
#include "Model/Local/CellComputerFunction.h"
#include "Model/Local/EnergyGuidance.h"
#include "Model/Local/CellFeatureFactory.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/ParticleMap.h"
#include "Model/Local/SpaceMetricLocal.h"
#include "Model/Local/ContextFactory.h"
#include "Model/Local/MapCompartment.h"
#include "Model/Local/UnitThreadController.h"
#include "Model/Local/UnitGrid.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/Unit.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/AccessPortFactory.h"
#include "Model/Local/SymbolTable.h"

#include "SimulationControllerImpl.h"
#include "CellComputerCompilerImpl.h"
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
	SimulationContextLocal* context = contextFactory->buildSimulationContext();

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
	controller->init(static_cast<SimulationContextLocal*>(context));

	return controller;
}

SimulationAccess * ModelBuilderFacadeImpl::buildSimulationAccess(SimulationContext * contextApi) const
{
	AccessPortFactory* factory = ServiceLocator::getInstance().getService<AccessPortFactory>();
	auto access = factory->buildSimulationAccess();
	access->init(contextApi);
	return access;
}

CellConnector * ModelBuilderFacadeImpl::buildCellConnector(SimulationContext* contextApi) const
{
	auto connector = new CellConnectorImpl();
	auto context = static_cast<SimulationContextLocal*>(contextApi);
	connector->init(context->getSpaceMetric(), context->getSimulationParameters(), context->getNumberGenerator());
	return connector;
}

CellComputerCompiler * ModelBuilderFacadeImpl::buildCellComputerCompiler(SimulationContext* context) const
{
	auto compiler = new CellComputerCompilerImpl();
	compiler->init(context->getSymbolTable(), context->getSimulationParameters());
	return compiler;
}

Unit * ModelBuilderFacadeImpl::buildSimulationUnit(IntVector2D gridPos, SimulationContextLocal* context) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto grid = context->getUnitGrid();
	auto threads = context->getUnitThreadController();

	auto unit = contextFactory->buildSimulationUnit();		//unit has no parent due to an QObject::moveToThread call later
	auto unitContext = contextFactory->buildSimulationUnitContext();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = static_cast<SpaceMetricLocal*>(context->getSpaceMetric())->clone();
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
