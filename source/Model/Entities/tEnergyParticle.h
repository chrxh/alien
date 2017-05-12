#ifndef ENERGYPARTICLE_H
#define ENERGYPARTICLE_H

#include <QVector2D>

#include "model/Definitions.h"
#include "model/Entities/Descriptions.h"
#include "Timestamp.h"

class EnergyParticle
	: public Timestamp
{
public:
	EnergyParticle(UnitContext* context) : Timestamp(context) {}
	virtual ~EnergyParticle() = default;

	virtual EnergyParticleDescription getDescription() const = 0;

    virtual bool processingMovement (CellCluster*& cluster) = 0;

	virtual qreal getEnergy() const = 0;
	virtual void setEnergy(qreal value) = 0;

	virtual QVector2D getPosition () const = 0;
	virtual void setPosition(QVector2D value) = 0;

	virtual QVector2D getVelocity() const = 0;
	virtual void setVelocity(QVector2D value) = 0;

	virtual quint64 getId() const = 0;
	virtual void setId(quint64 value) = 0;

	virtual EnergyParticleMetadata getMetadata() const = 0;
	virtual void setMetadata(EnergyParticleMetadata value) = 0;

    virtual void serializePrimitives (QDataStream& stream) const = 0;
	virtual void deserializePrimitives (QDataStream& stream) = 0;
};

#endif // ENERGYPARTICLE_H
