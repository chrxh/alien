#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/alienfacade.h"

class AlienFacadeImpl : public AlienFacade
{
public:
    AlienFacadeImpl ();
	~AlienFacadeImpl() = default;

	SimulationContext* buildSimulationContext() const override;
	Topology* buildTorusTopology() const override;

    CellCluster* buildCellCluster (SimulationUnitContext* context) const override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel
        , SimulationUnitContext* context) const override;

    Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, SimulationUnitContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
    Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, SimulationUnitContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const override;
    Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationUnitContext* context) const override;

	Token* buildToken(SimulationUnitContext* context, qreal energy) const override;

    CellTO buildFeaturedCellTO (Cell* cell) const override;
    void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, SimulationUnitContext* context) const override;
};

#endif // FACTORYFACADEIMPL_H
