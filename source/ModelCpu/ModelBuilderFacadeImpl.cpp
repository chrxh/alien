#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"

#include "ModelInterface/SimulationParameters.h"
#include "ModelInterface/SimulationAccess.h"
#include "ModelInterface/Settings.h"
#include "ModelInterface/SymbolTable.h"
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
#include "SpacePropertiesImpl.h"
#include "ContextFactory.h"
#include "MapCompartment.h"
#include "UnitThreadController.h"
#include "UnitGrid.h"
#include "SimulationContextImpl.h"
#include "Unit.h"
#include "UnitContext.h"
#include "AccessPortFactory.h"

#include "SimulationMonitorImpl.h"
#include "SimulationControllerImpl.h"
#include "CellComputerCompilerImpl.h"
#include "ModelBuilderFacadeImpl.h"
#include "DescriptionHelperImpl.h"
#include "SerializerImpl.h"

namespace
{
	const int ARRAY_SIZE_FOR_RANDOM_NUMBERS = 234327;
}

SimulationController* ModelBuilderFacadeImpl::buildSimulationController(int maxRunngingThreads, IntVector2D gridSize
	, IntVector2D universeSize, SymbolTable* symbolTable, SimulationParameters* parameters, uint timestep) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	SimulationContextImpl* context = contextFactory->buildSimulationContext();

	auto compiler = contextFactory->buildCellComputerCompiler();
	auto threads = contextFactory->buildSimulationThreads();
	auto grid = contextFactory->buildSimulationGrid();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = contextFactory->buildSpaceMetric();
	metric->init(universeSize);
	threads->init(maxRunngingThreads);
	grid->init(gridSize, metric);
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, 0);
	compiler->init(symbolTable, parameters);
	context->init(numberGen, metric, grid, threads, symbolTable, parameters, compiler);

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
	controller->init(static_cast<SimulationContextImpl*>(context), timestep);

	return controller;
}

SimulationAccess * ModelBuilderFacadeImpl::buildSimulationAccess() const
{
	AccessPortFactory* factory = ServiceLocator::getInstance().getService<AccessPortFactory>();
	return factory->buildSimulationAccess();;
}

SimulationMonitor * ModelBuilderFacadeImpl::buildSimulationMonitor() const
{
	return new SimulationMonitorImpl();
}

DescriptionHelper * ModelBuilderFacadeImpl::buildDescriptionHelper() const
{
	return new DescriptionHelperImpl();
}

Unit * ModelBuilderFacadeImpl::buildSimulationUnit(IntVector2D gridPos, SimulationContextImpl* context) const
{
	ContextFactory* contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();
	GlobalFactory* globalFactory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto grid = context->getUnitGrid();
	auto threads = context->getUnitThreadController();

	auto unit = contextFactory->buildSimulationUnit();		//unit has no parent due to an QObject::moveToThread call later
	auto unitContext = contextFactory->buildSimulationUnitContext();
	auto numberGen = globalFactory->buildRandomNumberGenerator();
	auto metric = static_cast<SpacePropertiesImpl*>(context->getSpaceProperties())->clone();
	auto compartment = contextFactory->buildMapCompartment();
	auto cellMap = contextFactory->buildCellMap();
	auto energyMap = contextFactory->buildEnergyParticleMap();
	auto parameters = context->getSimulationParameters()->clone();
	uint16_t threadId = gridPos.x + gridPos.y * grid->getSize().x + 1;
	numberGen->init(ARRAY_SIZE_FOR_RANDOM_NUMBERS, threadId);
	compartment->init(grid->calcCompartmentRect(gridPos));
	cellMap->init(metric, compartment);
	energyMap->init(metric, compartment);
	unitContext->init(numberGen, metric, cellMap, energyMap, compartment, parameters);
	unit->init(unitContext);

	return unit;
}

Serializer * ModelBuilderFacadeImpl::buildSerializer() const
{
	return new SerializerImpl();
}

SymbolTable * ModelBuilderFacadeImpl::buildDefaultSymbolTable() const
{
	return ModelSettings::getDefaultSymbolTable();
}

SimulationParameters* ModelBuilderFacadeImpl::buildDefaultSimulationParameters() const
{
	return ModelSettings::getDefaultSimulationParameters();
}
