#ifndef ENERGYPARTICLEMAPIMPL_H
#define ENERGYPARTICLEMAPIMPL_H

#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/MapCompartment.h"
#include "Model/Context/UnitContext.h"

class EnergyParticleMapImpl
	: public EnergyParticleMap
{
	Q_OBJECT
public:
	EnergyParticleMapImpl(QObject* parent = nullptr);
	virtual ~EnergyParticleMapImpl();

	virtual void init(SpaceMetric* metric, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void removeParticleIfPresent(QVector2D pos, Particle* particleToRemove) override;
	virtual void setParticle(QVector2D pos, Particle* particle) override;
	virtual Particle* getParticle(QVector2D pos) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream, QMap<quint64, Particle*> const& oldIdEnergyMap) override;

private:
	void deleteGrid();
	inline Particle*& locateParticle(IntVector2D & intPos) const;

	SpaceMetric* _metric = nullptr;
	MapCompartment* _compartment = nullptr;
	IntVector2D _size = { 0, 0 };
};

/****************** inline methods ******************/

Particle*& EnergyParticleMapImpl::locateParticle(IntVector2D & intPos) const
{
	if (_compartment->isPointInCompartment(intPos)) {
		_compartment->convertAbsToRelPosition(intPos);
		return _energyGrid[intPos.x][intPos.y];
	}
	else {
		auto energyMap = static_cast<EnergyParticleMapImpl*>(_compartment->getNeighborContext(intPos)->getEnergyParticleMap());
		_compartment->convertAbsToRelPosition(intPos);
		return energyMap->_energyGrid[intPos.x][intPos.y];
	}
}

#endif //ENERGYPARTICLEMAPIMPL_H