#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/factoryfacade.h"

class FactoryFacadeImpl : public FactoryFacade
{
public:
    FactoryFacadeImpl ();
	~FactoryFacadeImpl() = default;

    CellCluster* buildCellCluster (SimulationContext* context) override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
        , SimulationContext* context) override;
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, SimulationContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) override;
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, SimulationContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) override;
    Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationContext* context) override;
    CellTO buildFeaturedCellTO (Cell* cell) override;
    void changeFeaturesOfCell (Cell* cell, CellFunctionType type, SimulationContext* context) override;
};

#endif // FACTORYFACADEIMPL_H
