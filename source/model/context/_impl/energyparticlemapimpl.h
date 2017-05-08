#ifndef ENERGYPARTICLEMAPIMPL_H
#define ENERGYPARTICLEMAPIMPL_H

#include "model/context/EnergyParticleMap.h"
#include "model/context/MapCompartment.h"
#include "model/context/UnitContext.h"

class EnergyParticleMapImpl
	: public EnergyParticleMap
{
	Q_OBJECT
public:
	EnergyParticleMapImpl(QObject* parent = nullptr);
	virtual ~EnergyParticleMapImpl();

	virtual void init(SpaceMetric* metric, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void removeParticleIfPresent(QVector2D pos, EnergyParticle* particleToRemove) override;
	virtual void setParticle(QVector2D pos, EnergyParticle* particle) override;
	virtual EnergyParticle* getParticle(QVector2D pos) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream, QMap<quint64, EnergyParticle*> const& oldIdEnergyMap) override;

private:
	void deleteGrid();
	inline EnergyParticle*& locateParticle(IntVector2D const& intPos) const;

	SpaceMetric* _metric = nullptr;
	MapCompartment* _compartment = nullptr;
	IntVector2D _size = { 0, 0 };
};

/****************** inline methods ******************/

EnergyParticle*& EnergyParticleMapImpl::locateParticle(IntVector2D const& intPos) const
{
	if (_compartment->isPointInCompartment(intPos)) {
		auto relPos = _compartment->convertAbsToRelPosition(intPos);
		return _energyGrid[relPos.x][relPos.y];
	}
	else {
		auto energyMap = static_cast<EnergyParticleMapImpl*>(_compartment->getNeighborContext(intPos)->getEnergyParticleMap());
		auto relPos = _compartment->convertAbsToRelPosition(intPos);
		return energyMap->_energyGrid[relPos.x][relPos.y];
	}
}

#endif //ENERGYPARTICLEMAPIMPL_H