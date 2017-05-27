#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "Base/Definitions.h"
#include "model/Metadata/CellMetadata.h"
#include "model/Metadata/CellClusterMetadata.h"
#include "model/Metadata/EnergyParticleMetadata.h"

#include "DllExport.h"

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
struct CellClusterDescription;
struct CellDescription;
struct EnergyParticleDescription;
class SimulationAccess;
class QTimer;
class ModelBuilderFacade;
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


#endif // DEFINITIONS_H
