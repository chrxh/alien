#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/factoryfacade.h"

class FactoryFacadeImpl : public FactoryFacade
{
public:
    FactoryFacadeImpl ();
	~FactoryFacadeImpl() = default;

	SimulationContext* buildSimulationContext() const override;

    CellCluster* buildCellCluster (SimulationContext* context) const override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
        , SimulationContext* context) const override;

    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, SimulationContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, SimulationContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
    Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationContext* context) const override;

    CellTO buildFeaturedCellTO (Cell* cell) const override;
    void changeFeaturesOfCell (Cell* cell, CellFunctionType type, SimulationContext* context) const override;
};

#endif // FACTORYFACADEIMPL_H
