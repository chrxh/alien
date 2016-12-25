#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <QtGlobal>
#include <QVector3D>
#include <QSize>
#include <QMap>
#include <QSet>
#include <qmath.h>
#include <QDataStream>

#include <unordered_set>

class Cell;
class CellCluster;
class Grid;
class EnergyParticle;
class Token;
class CellFeature;
class SimulationUnit;
class CellMap;
class EnergyParticleMap;
class Topology;
class SimulationContext;

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

#endif // DEFINITIONS_H
