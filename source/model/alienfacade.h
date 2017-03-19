#ifndef FACTORYFACADE_H
#define FACTORYFACADE_H

#include <QtGlobal>
#include <QVector3D>

#include "model/entities/cellto.h"
#include "model/features/cellfeatureconstants.h"

#include "definitions.h"

class AlienFacade
{
public:
    virtual ~AlienFacade () {}

	virtual SimulationContext* buildSimulationContext() const = 0;

    virtual CellCluster* buildCellCluster (SimulationContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
        , QVector3D vel, SimulationContext* context) const = 0;

    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, QByteArray data, SimulationContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCell (qreal energy, Enums::CellFunction::Type type, SimulationContext* context
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildFeaturedCellWithRandomData (qreal energy, SimulationContext* context) const = 0;

    virtual CellTO buildFeaturedCellTO (Cell* cell) const = 0;
    virtual void changeFeaturesOfCell (Cell* cell, Enums::CellFunction::Type type, SimulationContext* context) const = 0;
};

#endif // FACTORYFACADE_H
