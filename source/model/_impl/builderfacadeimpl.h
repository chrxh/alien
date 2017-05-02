#ifndef BUILDERFACADEIMPL_H
#define BUILDERFACADEIMPL_H

#include "model/BuilderFacade.h"

class BuilderFacadeImpl
	: public BuilderFacade
{
public:
    BuilderFacadeImpl ();
	~BuilderFacadeImpl() = default;

	virtual SimulationContext* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric, SymbolTable* symbolTable
		, SimulationParameters* parameters, QObject* parent = nullptr) const override;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize, QObject* parent = nullptr) const override;

	virtual CellCluster* buildCellCluster (UnitContext* context) const override;
	virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
        , UnitContext* context) const override;

	virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, UnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, UnitContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCellWithRandomData (qreal energy, UnitContext* context) const override;

	virtual Token* buildToken(UnitContext* context, qreal energy) const override;

	virtual CellTO buildFeaturedCellTO (Cell* cell) const override;
	virtual void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const;
};

#endif // BUILDERFACADEIMPL_H
