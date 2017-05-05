#ifndef UNITGRIDIMPL_H
#define UNITGRIDIMPL_H

#include "model/context/UnitGrid.h"

class UnitGridImpl
	: public UnitGrid
{
	Q_OBJECT
public:
	UnitGridImpl(QObject* parent = nullptr);
	virtual ~UnitGridImpl() {}

	virtual void init(IntVector2D gridSize, SpaceMetric* metric) override;

	virtual void registerUnit(IntVector2D gridPos, Unit* unit) override;
	virtual IntVector2D getSize() const override;
	virtual Unit* getUnitOfGridPos(IntVector2D gridPos) const override;
	virtual Unit* getUnitOfMapPos(QVector3D pos) const override;
	virtual IntRect calcCompartmentRect(IntVector2D gridPos) const override;

private:
	IntVector2D calcCompartmentSize() const;

	SpaceMetric* _metric = nullptr;
	IntVector2D _gridSize = { 0, 0 };
	std::vector<std::vector<Unit*>> _units;
};

#endif // UNITGRIDIMPL_H
