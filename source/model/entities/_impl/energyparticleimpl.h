#ifndef ENERGYPARTICLEIMPL_H
#define ENERGYPARTICLEIMPL_H

#include "model/entities/EnergyParticle.h"
#include "model/entities/Descriptions.h"

class EnergyParticleImpl
	: public EnergyParticle
{
public:
	EnergyParticleImpl(UnitContext* context);
	EnergyParticleImpl(qreal energy, QVector2D pos, QVector2D vel, UnitContext* context);

	virtual void init(UnitContext* context) override;

	bool processingMovement(CellCluster*& cluster) override;

	virtual qreal getEnergy() const override;
	virtual void setEnergy(qreal value) override;

	virtual QVector2D getPosition() const override;
	virtual void setPosition(QVector2D value) override;

	virtual QVector2D getVelocity() const override;
	virtual void setVelocity(QVector2D value) override;

	virtual quint64 getId() const override;
	virtual void setId(quint64 value) override;

	virtual EnergyParticleMetadata getMetadata() const override;
	virtual void setMetadata(EnergyParticleMetadata value) override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

private:
	void move();
	void collisionWithEnergyParticle(EnergyParticle* otherEnergy);
	void collisionWithCell(Cell* cell);

	CellDescription getRandomCellDesciption(double energy) const;

	UnitContext* _context = nullptr;

	qreal _energy = 0.0;
	QVector2D _pos;
	QVector2D _vel;
	quint64 _id = 0;
	EnergyParticleMetadata _metadata;
};

#endif // ENERGYPARTICLEIMPL_H
