#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"

#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SpaceProperties.h"

#include "Cell.h"
#include "Cluster.h"
#include "Particle.h"
#include "Token.h"
#include "EntityFactory.h"
#include "CellFunction.h"
#include "CellComputerFunction.h"
#include "EnergyGuidance.h"
#include "CellFeatureFactory.h"
#include "CellMap.h"
#include "ParticleMap.h"
#include "ContextFactory.h"
#include "MapCompartment.h"
#include "UnitThreadController.h"
#include "UnitGrid.h"
#include "SimulationContextCpuImpl.h"
#include "Unit.h"
#include "UnitContext.h"
#include "AccessPortFactory.h"
#include "ModelCpuData.h"

#include "SimulationMonitorImpl.h"
#include "SimulationControllerCpuImpl.h"
#include "CellComputerCompilerImpl.h"
#include "ModelCpuBuilderFacadeImpl.h"

namespace
{
	const int ARRAY_SIZE_FOR_RANDOM_NUMBERS = 234327;
}

SimulationControllerCpu * ModelCpuBuilderFacadeImpl::buildSimulationController(Config const & config
	, ModelCpuData const& specificData
	, uint timestepAtBeginning) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	SimulationContextCpuImpl* context = contextFactory->buildSimulationContext();

	auto compiler = contextFactory->buildCellComputerCompiler();
	auto threads = contextFactory->buildSimulationThreads();
	auto grid = contextFactory->buildSimulationGrid();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto spaceProp = new SpaceProperties();

	IntVector2D gridSize = specificData.getGridSize();
	spaceProp->init(config.universeSize);
	threads->init(specificData.getMaxRunningThreads());
	grid->init(gridSize, spaceProp);
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, 0);
	compiler->init(config.symbolTable, config.parameters);
	context->init(numberGen, spaceProp, grid, threads, config.symbolTable, config.parameters, compiler);

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

	auto controller = new SimulationControllerCpuImpl();
	controller->init(static_cast<SimulationContextCpuImpl*>(context), timestepAtBeginning);

	return controller;
}

SimulationAccessCpu * ModelCpuBuilderFacadeImpl::buildSimulationAccess() const
{
	AccessPortFactory* factory = ServiceLocator::getInstance().getService<AccessPortFactory>();
	return factory->buildSimulationAccess();;
}

SimulationMonitor * ModelCpuBuilderFacadeImpl::buildSimulationMonitor() const
{
	return new SimulationMonitorImpl();
}

Unit * ModelCpuBuilderFacadeImpl::buildSimulationUnit(IntVector2D gridPos, SimulationContextCpuImpl* context) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto grid = context->getUnitGrid();
	auto threads = context->getUnitThreadController();

	auto unit = contextFactory->buildSimulationUnit();		//unit has no parent due to an QObject::moveToThread call later
	auto unitContext = contextFactory->buildSimulationUnitContext();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto spaceProp = context->getSpaceProperties()->clone();
	auto compartment = contextFactory->buildMapCompartment();
	auto cellMap = contextFactory->buildCellMap();
	auto energyMap = contextFactory->buildEnergyParticleMap();
	auto parameters = context->getSimulationParameters()->clone();
	uint16_t threadId = gridPos.x + gridPos.y * grid->getSize().x + 1;
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, threadId);
	compartment->init(grid->calcCompartmentRect(gridPos));
	cellMap->init(spaceProp, compartment);
	energyMap->init(spaceProp, compartment);
	unitContext->init(numberGen, spaceProp, cellMap, energyMap, compartment, parameters);
	unit->init(unitContext);

	return unit;
}

