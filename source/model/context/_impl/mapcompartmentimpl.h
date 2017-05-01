#ifndef MAPCOMPARTMENTIMPL_H
#define MAPCOMPARTMENTIMPL_H

#include "model/context/mapcompartment.h"

class MapCompartmentImpl
	: public MapCompartment
{
	Q_OBJECT
public:
	MapCompartmentImpl(QObject* parent = nullptr);
	virtual ~MapCompartmentImpl() {}

	virtual void init(SpaceMetric* metric, IntRect mapRect) override;

	virtual void registerNeighborContext(RelativeLocation location, SimulationUnitContext* context) override;
	virtual SimulationUnitContext* getNeighborContext(RelativeLocation location) const override;
	virtual SimulationUnitContext* getNeighborContext(IntVector2D pos) const override;

private:
};

#endif // MAPCOMPARTMENTIMPL_H
