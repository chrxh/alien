#include "global/ServiceLocator.h"

#include "EnergyParticleImpl.h"
#include "CellImpl.h"
#include "CellClusterImpl.h"
#include "TokenImpl.h"
#include "EntityFactoryImpl.h"

namespace
{
    EntityFactoryImpl instance;
}

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildCellCluster (UnitContext* context) const
{
    return new CellClusterImpl(context);
}

CellCluster* EntityFactoryImpl::buildCellCluster (QList< Cell* > cells, qreal angle
    , QVector3D pos, qreal angularVel, QVector3D vel, UnitContext* context) const
{
    return new CellClusterImpl(cells, angle, pos, angularVel, vel, context);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells
    , qreal angle, UnitContext* context) const
{
    return new CellClusterImpl(cells, angle, context);
}

Cell* EntityFactoryImpl::buildCell (UnitContext* context) const
{
    return new CellImpl(context);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, UnitContext* context, int maxConnections
    , int tokenAccessNumber, QVector3D relPos) const
{
    return new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
}

Token* EntityFactoryImpl::buildToken (UnitContext* context) const
{
    return new TokenImpl(context);
}

Token * EntityFactoryImpl::buildToken(UnitContext* context, qreal energy) const
{
	return new TokenImpl(context, energy);
}

Token * EntityFactoryImpl::buildToken(UnitContext* context, qreal energy, QByteArray const& memory) const
{
	return new TokenImpl(context, energy, memory);
}

Token * EntityFactoryImpl::buildTokenWithRandomData(UnitContext* context, qreal energy) const
{
	return new TokenImpl(context, energy, true);
}

EnergyParticle* EntityFactoryImpl::buildEnergyParticle(UnitContext* context) const
{
    return new EnergyParticleImpl(context);
}

EnergyParticle *EntityFactoryImpl::buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
    , UnitContext *context) const
{
    return new EnergyParticleImpl(energy, pos, vel, context);
}
