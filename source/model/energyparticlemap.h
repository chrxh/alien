#ifndef ENERGYPARTICLEMAP_H
#define ENERGYPARTICLEMAP_H

#include "definitions.h"

class EnergyParticleMap
{
public:
	EnergyParticleMap();
	virtual ~EnergyParticleMap();

	void init(Topology* topo);
	void clear();
	
	void removeParticleIfPresent(QVector3D pos, EnergyParticle* energy);
	void setParticle(QVector3D pos, EnergyParticle* energy);
	EnergyParticle* getParticle(QVector3D pos) const;
	inline EnergyParticle* getParticleFast(IntVector2D const& pos) const;

	void serializePrimitives(QDataStream& stream) const;
	void deserializePrimitives(QDataStream& stream, QMap<quint64, EnergyParticle*> const& oldIdEnergyMap);

private:
	void deleteGrid();

	Topology* _topo = nullptr;
	EnergyParticle*** _energyGrid = nullptr;
	int _gridSize = 0;
};

EnergyParticle * EnergyParticleMap::getParticleFast(IntVector2D const& intPos) const
{
	return _energyGrid[intPos.x][intPos.y];
}

#endif //ENERGYPARTICLEMAP_H