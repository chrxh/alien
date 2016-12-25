#include "entityfactoryimpl.h"

#include "cellimpl.h"
#include "cellclusterimpl.h"
#include "model/entities/token.h"

#include "global/servicelocator.h"

namespace
{
    EntityFactoryImpl entityFactoryImpl;
}

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildEmptyCellCluster (SimulationContext* context)
{
    return new CellClusterImpl(context);
}

CellCluster* EntityFactoryImpl::buildCellCluster (QList< Cell* > cells, qreal angle
    , QVector3D pos, qreal angularVel, QVector3D vel, SimulationContext* context)
{
    return new CellClusterImpl(cells, angle, pos, angularVel, vel, context);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells
    , qreal angle, SimulationContext* context)
{
    return new CellClusterImpl(cells, angle, context);
}

Cell* EntityFactoryImpl::buildEmptyCell (SimulationContext* context) override
{
    return new CellImpl(context);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, SimulationContext* context, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    return new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
}

Cell* EntityFactoryImpl::buildCellWithRandomData (qreal energy, SimulationContext* context)
{
    return new CellImpl(energy, context, true);
}

Token* EntityFactoryImpl::buildEmptyToken ()
{
    return new Token();
}
