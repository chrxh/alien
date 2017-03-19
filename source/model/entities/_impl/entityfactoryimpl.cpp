#include "global/servicelocator.h"

#include "model/entities/energyparticle.h"

#include "cellimpl.h"
#include "cellclusterimpl.h"
#include "tokenimpl.h"
#include "entityfactoryimpl.h"

namespace
{
    EntityFactoryImpl entityFactoryImpl;
}

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildCellCluster (SimulationContext* context) const
{
    return new CellClusterImpl(context);
}

CellCluster* EntityFactoryImpl::buildCellCluster (QList< Cell* > cells, qreal angle
    , QVector3D pos, qreal angularVel, QVector3D vel, SimulationContext* context) const
{
    return new CellClusterImpl(cells, angle, pos, angularVel, vel, context);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells
    , qreal angle, SimulationContext* context) const
{
    return new CellClusterImpl(cells, angle, context);
}

Cell* EntityFactoryImpl::buildCell (SimulationContext* context) const
{
    return new CellImpl(context);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, SimulationContext* context, int maxConnections
    , int tokenAccessNumber, QVector3D relPos) const
{
    return new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
}

Token* EntityFactoryImpl::buildToken (SimulationContext* context) const
{
    return new TokenImpl(context);
}

Token * EntityFactoryImpl::buildToken(SimulationContext* context, qreal energy) const
{
	return new TokenImpl(context, energy);
}

Token * EntityFactoryImpl::buildToken(SimulationContext* context, qreal energy, QByteArray const& memory) const
{
	return new TokenImpl(context, energy, memory);
}

Token * EntityFactoryImpl::buildTokenWithRandomData(SimulationContext* context, qreal energy) const
{
	return new TokenImpl(context, energy, true);
}

EnergyParticle* EntityFactoryImpl::buildEnergyParticle(SimulationContext* context) const
{
    return new EnergyParticle(context);
}

EnergyParticle *EntityFactoryImpl::buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
    , SimulationContext *context) const
{
    return new EnergyParticle(energy, pos, vel, context);
}
