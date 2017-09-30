#pragma once

#include "Model/Local/CellFeatureFactory.h"

class CellFeatureFactoryImpl
	: public CellFeatureFactory
{
public:
    ~CellFeatureFactoryImpl () {}

    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const override;
    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, UnitContext* context) const override;

    CellFeature* addEnergyGuidance (Cell* cell, UnitContext* context) const override;
};
