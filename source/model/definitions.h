#ifndef MODEL_DEFINITIONS_H
#define MODEL_DEFINITIONS_H

#include "Base/Definitions.h"
#include "model/metadata/CellMetadata.h"
#include "model/metadata/CellClusterMetadata.h"
#include "model/metadata/EnergyParticleMetadata.h"

class Cell;
class CellCluster;
class UnitGrid;
class EnergyParticle;
class Token;
class CellFeature;
class Unit;
class CellMap;
class EnergyParticleMap;
class SpaceMetric;
class MapCompartment;
class UnitGrid;
class UnitThreadController;
class UnitContext;
class SimulationContext;
class SimulationContextApi;
class SimulationParameters;
class SimulationController;
class SymbolTable;
class UnitObserver;
class EntityFactory;
class ContextFactory;
struct DataDescription;
struct DataLightDescription;
class SimulationAccess;

class BuilderFacade;
class SerializationFacade;

struct CellClusterHash
{
	std::size_t operator()(CellCluster* const& s) const;
};
typedef std::unordered_set<CellCluster*, CellClusterHash> CellClusterSet;

struct CellHash
{
    std::size_t operator()(Cell* const& s) const;
};
typedef std::unordered_set<Cell*, CellHash> CellSet;


#endif // MODEL_DEFINITIONS_H
