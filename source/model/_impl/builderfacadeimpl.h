#ifndef BUILDERFACADEIMPL_H
#define BUILDERFACADEIMPL_H

#include "model/BuilderFacade.h"

class BuilderFacadeImpl
	: public BuilderFacade
{
public:
    BuilderFacadeImpl ();
	virtual ~BuilderFacadeImpl() = default;

	virtual SimulationAccessApi* buildSimulationManipulator(SimulationContextApi* context) const override;
	virtual SimulationController* buildSimulationController(SimulationContextApi* context) const override;
	virtual SimulationContextApi* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric
		, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize) const override;
	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

	virtual CellCluster* buildCellCluster(UnitContext* context) const override;
	virtual CellCluster* buildCellCluster(QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
		, UnitContext* context) const override;

	virtual Cell* buildFeaturedCell(qreal energy, Enums::CellFunction::Type type, QByteArray data, UnitContext* context
		, int maxConnections = 0, int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCell(qreal energy, Enums::CellFunction::Type type, UnitContext* context
		, int maxConnections = 0, int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const override;
	virtual Cell* buildFeaturedCellWithRandomData(qreal energy, UnitContext* context) const override;
	virtual CellTO buildFeaturedCellTO(Cell* cell) const override;

	virtual Token* buildToken(UnitContext* context, qreal energy) const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const;
};

#endif // BUILDERFACADEIMPL_H
