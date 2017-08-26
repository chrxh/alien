#ifndef ENERGYPARTICLEMAP_H
#define ENERGYPARTICLEMAP_H

#include "Model/Definitions.h"

class EnergyParticleMap
	: public QObject
{
	Q_OBJECT
public:
	EnergyParticleMap(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~EnergyParticleMap() = default;

	virtual void init(SpaceMetric* topo, MapCompartment* compartment) = 0;
	virtual void clear() = 0;
	
	virtual void removeParticleIfPresent(QVector2D pos, Particle* particleToRemove) = 0;
	virtual void setParticle(QVector2D pos, Particle* particle) = 0;
	virtual Particle* getParticle(QVector2D pos) const = 0;
	inline Particle* getParticleFast(IntVector2D const& pos) const;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream, QMap<quint64, Particle*> const& oldIdEnergyMap) = 0;

protected:
	Particle*** _energyGrid = nullptr;
};

Particle * EnergyParticleMap::getParticleFast(IntVector2D const& intPos) const
{
	return _energyGrid[intPos.x][intPos.y];
}

#endif //ENERGYPARTICLEMAP_H