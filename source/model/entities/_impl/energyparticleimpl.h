#ifndef ENERGYPARTICLEIMPL_H
#define ENERGYPARTICLEIMPL_H

#include "model/entities/energyparticle.h"

class EnergyParticleImpl
	: public EnergyParticle
{
public:
	EnergyParticleImpl(SimulationContext* context);
	EnergyParticleImpl(qreal energy, QVector3D pos, QVector3D vel, SimulationContext* context);

	bool processingMovement(CellCluster*& cluster) override;

	virtual qreal getEnergy() const override;
	virtual void setEnergy(qreal value) override;

	virtual QVector3D getPosition() const override;
	virtual void setPosition(QVector3D value) override;

	virtual QVector3D getVelocity() const override;
	virtual void setVelocity(QVector3D value) override;

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

	SimulationContext* _context = nullptr;
	Topology* _topology = nullptr;
	CellMap* _cellMap = nullptr;
	EnergyParticleMap* _energyMap = nullptr;
	SimulationParameters* _parameters = nullptr;

	qreal _energy = 0.0;
	QVector3D _pos;
	QVector3D _vel;
	quint64 _id = 0;
	EnergyParticleMetadata _metadata;
};

#endif // ENERGYPARTICLEIMPL_H
