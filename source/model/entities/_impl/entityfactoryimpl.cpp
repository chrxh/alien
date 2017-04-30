#include "global/servicelocator.h"

#include "energyparticleimpl.h"
#include "cellimpl.h"
#include "cellclusterimpl.h"
#include "tokenimpl.h"
#include "entityfactoryimpl.h"

namespace
{
    EntityFactoryImpl instance;
}

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildCellCluster (SimulationUnitContext* context) const
{
    return new CellClusterImpl(context);
}

CellCluster* EntityFactoryImpl::buildCellCluster (QList< Cell* > cells, qreal angle
    , QVector3D pos, qreal angularVel, QVector3D vel, SimulationUnitContext* context) const
{
    return new CellClusterImpl(cells, angle, pos, angularVel, vel, context);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells
    , qreal angle, SimulationUnitContext* context) const
{
    return new CellClusterImpl(cells, angle, context);
}

Cell* EntityFactoryImpl::buildCell (SimulationUnitContext* context) const
{
    return new CellImpl(context);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, SimulationUnitContext* context, int maxConnections
    , int tokenAccessNumber, QVector3D relPos) const
{
    return new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
}

Token* EntityFactoryImpl::buildToken (SimulationUnitContext* context) const
{
    return new TokenImpl(context);
}

Token * EntityFactoryImpl::buildToken(SimulationUnitContext* context, qreal energy) const
{
	return new TokenImpl(context, energy);
}

Token * EntityFactoryImpl::buildToken(SimulationUnitContext* context, qreal energy, QByteArray const& memory) const
{
	return new TokenImpl(context, energy, memory);
}

Token * EntityFactoryImpl::buildTokenWithRandomData(SimulationUnitContext* context, qreal energy) const
{
	return new TokenImpl(context, energy, true);
}

EnergyParticle* EntityFactoryImpl::buildEnergyParticle(SimulationUnitContext* context) const
{
    return new EnergyParticleImpl(context);
}

EnergyParticle *EntityFactoryImpl::buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
    , SimulationUnitContext *context) const
{
    return new EnergyParticleImpl(energy, pos, vel, context);
}
