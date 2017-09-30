#pragma once

#include "Model/Local/Particle.h"
#include "Model/Api/Descriptions.h"

class ParticleImpl
	: public Particle
{
public:
	ParticleImpl(uint64_t id, qreal energy, QVector2D pos, QVector2D vel, UnitContext* context);

	virtual ParticleDescription getDescription() const override;
	virtual void applyChangeDescription(ParticleChangeDescription const& change) override;

	virtual bool processingMovement(Cluster*& cluster) override;

	virtual qreal getEnergy() const override;
	virtual void setEnergy(qreal value) override;

	virtual QVector2D getPosition() const override;
	virtual void setPosition(QVector2D value) override;

	virtual QVector2D getVelocity() const override;
	virtual void setVelocity(QVector2D value) override;

	virtual quint64 getId() const override;
	virtual void setId(quint64 value) override;

	virtual ParticleMetadata getMetadata() const override;
	virtual void setMetadata(ParticleMetadata value) override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

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

