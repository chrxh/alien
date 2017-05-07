#ifndef MODEL_DEFINITIONS_H
#define MODEL_DEFINITIONS_H

#include "global/Definitions.h"
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
class EntityFactory;
class ContextFactory;
struct DataDescription;
struct DataLightDescription;
template<typename DataDescriptionType>
class SimulationAccess;
using SimulationFullAccess = SimulationAccess<DataDescription>;
using SimulationLightAccess = SimulationAccess<DataLightDescription>;

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

struct IntVector2D {
	int x;
	int y;
};

extern bool operator==(IntVector2D const& vec1, IntVector2D const& vec2);
extern std::ostream& operator << (std::ostream& os, const IntVector2D& vec);

struct IntRect {
	IntVector2D p1;
	IntVector2D p2;
};


#endif // MODEL_DEFINITIONS_H
