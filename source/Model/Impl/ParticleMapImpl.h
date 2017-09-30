#pragma once

#include "Model/Local/ParticleMap.h"
#include "Model/Local/MapCompartment.h"
#include "Model/Local/UnitContext.h"

class ParticleMapImpl
	: public ParticleMap
{
	Q_OBJECT
public:
	ParticleMapImpl(QObject* parent = nullptr);
	virtual ~ParticleMapImpl();

	virtual void init(SpaceMetricLocal* metric, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void removeParticleIfPresent(QVector2D pos, Particle* particleToRemove) override;
	virtual void setParticle(QVector2D pos, Particle* particle) override;
	virtual Particle* getParticle(QVector2D pos) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream, QMap<quint64, Particle*> const& oldIdEnergyMap) override;

private:
	void deleteGrid();
	inline Particle*& locateParticle(IntVector2D & intPos) const;

	SpaceMetricLocal* _metric = nullptr;
	MapCompartment* _compartment = nullptr;
	IntVector2D _size = { 0, 0 };
};

/****************** inline methods ******************/

Particle*& ParticleMapImpl::locateParticle(IntVector2D & intPos) const
{
	if (_compartment->isPointInCompartment(intPos)) {
		_compartment->convertAbsToRelPosition(intPos);
		return _energyGrid[intPos.x][intPos.y];
	}
	else {
		auto energyMap = static_cast<ParticleMapImpl*>(_compartment->getNeighborContext(intPos)->getEnergyParticleMap());
		_compartment->convertAbsToRelPosition(intPos);
		return energyMap->_energyGrid[intPos.x][intPos.y];
	}
}
