#ifndef ENERGYPARTICLEMAPIMPL_H
#define ENERGYPARTICLEMAPIMPL_H

#include "model/context/energyparticlemap.h"

class EnergyParticleMapImpl
	: public EnergyParticleMap
{
	Q_OBJECT
public:
	EnergyParticleMapImpl(QObject* parent = nullptr);
	virtual ~EnergyParticleMapImpl();

	virtual void init(Topology* topo, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void removeParticleIfPresent(QVector3D pos, EnergyParticle* energy) override;
	virtual void setParticle(QVector3D pos, EnergyParticle* energy) override;
	virtual EnergyParticle* getParticle(QVector3D pos) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream, QMap<quint64, EnergyParticle*> const& oldIdEnergyMap) override;

private:
	void deleteGrid();

	Topology* _topo = nullptr;
	int _gridSize = 0;
};

#endif //ENERGYPARTICLEMAPIMPL_H