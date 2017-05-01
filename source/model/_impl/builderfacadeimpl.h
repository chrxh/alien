#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/builderfacade.h"

class BuilderFacadeImpl
	: public BuilderFacade
{
public:
    BuilderFacadeImpl ();
	~BuilderFacadeImpl() = default;

	virtual SimulationContext* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, Topology* topology, SymbolTable* symbolTable
		, SimulationParameters* parameters, QObject* parent = nullptr) const override;
	virtual Topology* buildTorusTopology(IntVector2D universeSize, QObject* parent = nullptr) const override;

	virtual CellCluster* buildCellCluster (SimulationUnitContext* context) const override;
	virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
        , SimulationUnitContext* context) const override;

	virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, SimulationUnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, SimulationUnitContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationUnitContext* context) const override;

	virtual Token* buildToken(SimulationUnitContext* context, qreal energy) const override;

	virtual CellTO buildFeaturedCellTO (Cell* cell) const override;
	virtual void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, SimulationUnitContext* context) const override;

private:
	SimulationUnit* buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const;
};

#endif // FACTORYFACADEIMPL_H
