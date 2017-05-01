#ifndef FACTORYFACADE_H
#define FACTORYFACADE_H

#include <QtGlobal>
#include <QVector3D>

#include "model/entities/cellto.h"
#include "model/features/cellfeatureconstants.h"

#include "definitions.h"

class BuilderFacade
{
public:
    virtual ~BuilderFacade () {}

	virtual SimulationContext* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric, SymbolTable* symbolTable
		, SimulationParameters* parameters, QObject* parent = nullptr) const = 0;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize, QObject* parent = nullptr) const = 0;

    virtual CellCluster* buildCellCluster (SimulationUnitContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
        , QVector3D vel, SimulationUnitContext* context) const = 0;

    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, SimulationUnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, SimulationUnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationUnitContext* context) const = 0;

	virtual Token* buildToken(SimulationUnitContext* context, qreal energy) const = 0;


    virtual CellTO buildFeaturedCellTO (Cell* cell) const = 0;
    virtual void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, SimulationUnitContext* context) const = 0;
};

#endif // FACTORYFACADE_H
