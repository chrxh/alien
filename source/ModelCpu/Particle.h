#pragma once

#include "ModelBasic/Definitions.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/Descriptions.h"
#include "EntityWithTimestamp.h"

class Particle
	: public EntityWithTimestamp
{
public:
	Particle(uint64_t id, qreal energy, QVector2D pos, QVector2D vel, UnitContext* context);

	virtual ParticleDescription getDescription(ResolveDescription const& resolveDescription) const;
	virtual void applyChangeDescription(ParticleChangeDescription const& change);

	virtual bool processingMovement(Cluster*& cluster);

	virtual void clearParticleFromMap();
	virtual void drawParticleToMap();

	virtual qreal getEnergy() const;
	virtual void setEnergy(qreal value);

	virtual QVector2D getPosition() const;
	virtual void setPosition(QVector2D value);

	virtual QVector2D getVelocity() const;
	virtual void setVelocity(QVector2D value);

	virtual quint64 getId() const;
	virtual void setId(quint64 value);

	virtual ParticleMetadata getMetadata() const;
	virtual void setMetadata(ParticleMetadata value);

private:
	void move();
	void collisionWithEnergyParticle(Particle* otherEnergy);
	void collisionWithCell(Cell* cell);

	CellDescription getRandomCellDesciption(double energy) const;

	quint64 _id = 0;
	qreal _energy = 0.0;
	QVector2D _pos;
	QVector2D _vel;
	ParticleMetadata _metadata;
};

