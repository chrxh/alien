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

struct IntVector2D {
	int x;
	int y;

	IntVector2D() : x(0), y(0) { }
	IntVector2D(std::initializer_list<int> l)
	{
		auto it = l.begin();
		x = *it++;
		y = *it;
	}
	IntVector2D(QVector2D const& vec) : x(static_cast<int>(vec.x())), y(static_cast<int>(vec.y())) { }
	QVector2D toQVector2D()	{ return QVector2D(x, y); }
};

extern bool operator==(IntVector2D const& vec1, IntVector2D const& vec2);
extern std::ostream& operator << (std::ostream& os, const IntVector2D& vec);

struct IntRect {
	IntVector2D p1;
	IntVector2D p2;

	bool isContained(IntVector2D p)
	{
		return p1.x <= p.x && p1.y <= p.y && p.x <= p2.x && p.y <= p2.y;
	}
};


#endif // MODEL_DEFINITIONS_H
