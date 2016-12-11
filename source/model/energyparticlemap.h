#ifndef ENERGYPARTICLEMAP_H
#define ENERGYPARTICLEMAP_H

#include "definitions.h"

class EnergyParticleMap
{
public:
	EnergyParticleMap(Topology* topo);
	virtual ~EnergyParticleMap();

	void topologyUpdated();
	void clear();
	
	void removeParticleIfPresent(QVector3D pos, EnergyParticle* energy);
	void setParticle(QVector3D pos, EnergyParticle* energy);
	EnergyParticle* getParticle(QVector3D pos) const;
	inline EnergyParticle* getParticleFast(IntVector2D const& pos) const;

	void serialize(QDataStream& stream) const;
	void deserialize(QDataStream& stream, QMap<quint64, EnergyParticle*> const& oldIdEnergyMap);

private:
	void deleteCellMap();

	Topology* _topo = nullptr;
	EnergyParticle*** _energyGrid = nullptr;
};

EnergyParticle * EnergyParticleMap::getParticleFast(IntVector2D const& intPos) const
{
	return _energyGrid[intPos.x][intPos.y];
}

#endif //ENERGYPARTICLEMAP_H