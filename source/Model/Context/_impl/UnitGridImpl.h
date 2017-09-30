#pragma once

#include "Model/Context/UnitGrid.h"

class UnitGridImpl
	: public UnitGrid
{
	Q_OBJECT
public:
	UnitGridImpl(QObject* parent = nullptr);
	virtual ~UnitGridImpl();

	virtual void init(IntVector2D gridSize, SpaceMetricLocal* metric) override;

	virtual void registerUnit(IntVector2D gridPos, Unit* unit) override;
	virtual IntVector2D getSize() const override;
	virtual Unit* getUnitOfGridPos(IntVector2D gridPos) const override;
	virtual Unit* getUnitOfMapPos(QVector2D pos, CorrectionMode mode = CorrectionMode::Torus) const override;
	virtual IntVector2D getGridPosOfMapPos(QVector2D pos, CorrectionMode mode = CorrectionMode::Torus) const override;
	virtual IntRect calcCompartmentRect(IntVector2D gridPos) const override;

private:
	IntVector2D calcCompartmentSize() const;

	SpaceMetricLocal* _metric = nullptr;
	IntVector2D _gridSize = { 0, 0 };
	vector<vector<Unit*>> _units;
};

