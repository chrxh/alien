#pragma once

#include "Base/Definitions.h"
#include "ModelBasic/Definitions.h"

#include "DllExport.h"

class QTimer;

class UnitGrid;
class Particle;
class Token;
class CellFeatureChain;
class Unit;
class CellMap;
class ParticleMap;
class SpaceProperties;
class MapCompartment;
class UnitGrid;
class UnitThreadController;
class UnitContext;
class SimulationContextCpuImpl;
class SimulationContext;
class SimulationParameters;
class SimulationController;
class SymbolTable;
class UnitObserver;
class EntityFactory;
class ContextFactory;
class SimulationAccessCpu;
class DescriptionHelper;
class CellComputerCompiler;
class SimulationMonitorCpu;
class ModelCpuData;
class SimulationControllerCpu;
class ModelCpuBuilderFacade;

class CellComputerCompilerImpl;
class Cell;
class Cluster;
class UnitThread;
class SimulationAttributeSetter;

struct CellClusterHash
{
	std::size_t operator()(Cluster* const& s) const;
};
typedef std::unordered_set<Cluster*, CellClusterHash> CellClusterSet;

struct CellHash
{
	std::size_t operator()(Cell* const& s) const;
};
typedef std::unordered_set<Cell*, CellHash> CellSet;
