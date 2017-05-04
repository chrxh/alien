#ifndef FACTORYFACADE_H
#define FACTORYFACADE_H

#include <QtGlobal>
#include <QVector3D>

#include "model/entities/CellTO.h"
#include "model/features/CellFeatureEnums.h"

#include "Definitions.h"

class BuilderFacade
{
public:
	virtual ~BuilderFacade() = default;

	virtual SimulationController* buildSimulationController(SimulationContext* context) const = 0;
	virtual SimulationContext* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric
		, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize) const = 0;

    virtual CellCluster* buildCellCluster (UnitContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
		, UnitContext* context) const = 0;

    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, UnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, UnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCellWithRandomData (qreal energy, UnitContext* context) const = 0;
    virtual CellTO buildFeaturedCellTO (Cell* cell) const = 0;
    virtual void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const = 0;

	virtual Token* buildToken(UnitContext* context, qreal energy) const = 0;
};

#endif // FACTORYFACADE_H
