#pragma once

#include "Base/Definitions.h"
#include "ModelInterface/Definitions.h"

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
class SpacePropertiesImpl;
class MapCompartment;
class UnitGrid;
class UnitThreadController;
class UnitContext;
class SimulationContextImpl;
class SimulationContext;
class SimulationParameters;
class SimulationController;
class SymbolTable;
class UnitObserver;
class EntityFactory;
class ContextFactory;
struct DataChangeDescription;
struct ClusterChangeDescription;
struct CellChangeDescription;
struct ParticleChangeDescription;
struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;
struct CellFeatureDescription;
class SimulationAccess;
class ModelBuilderFacade;
class SerializationFacade;
class DescriptionHelper;
class CellComputerCompiler;
class Serializer;
class SimulationMonitor;

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
