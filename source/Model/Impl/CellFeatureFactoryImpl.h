#pragma once

#include "Model/Local/CellFeatureFactory.h"

class CellFeatureFactoryImpl
	: public CellFeatureFactory
{
public:
    ~CellFeatureFactoryImpl () {}

    CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const override;
    CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, UnitContext* context) const override;

    CellFeatureChain* addEnergyGuidance (Cell* cell, UnitContext* context) const override;
};
