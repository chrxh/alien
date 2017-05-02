#ifndef ENERGYPARTICLE_H
#define ENERGYPARTICLE_H

#include <QVector3D>

#include "model/Definitions.h"

class EnergyParticle
{
public:
	virtual ~EnergyParticle() {}

    virtual bool processingMovement (CellCluster*& cluster) = 0;

	virtual qreal getEnergy() const = 0;
	virtual void setEnergy(qreal value) = 0;

	virtual QVector3D getPosition () const = 0;
	virtual void setPosition(QVector3D value) = 0;

	virtual QVector3D getVelocity() const = 0;
	virtual void setVelocity(QVector3D value) = 0;

	virtual quint64 getId() const = 0;
	virtual void setId(quint64 value) = 0;

	virtual EnergyParticleMetadata getMetadata() const = 0;
	virtual void setMetadata(EnergyParticleMetadata value) = 0;

    virtual void serializePrimitives (QDataStream& stream) const = 0;
	virtual void deserializePrimitives (QDataStream& stream) = 0;
};

#endif // ENERGYPARTICLE_H
