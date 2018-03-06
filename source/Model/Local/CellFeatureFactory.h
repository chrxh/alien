#pragma once

#include <QString>

#include "Model/Api/CellFeatureEnums.h"
#include "Definitions.h"

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, UnitContext* context) const = 0;
    virtual CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const = 0;

    virtual CellFeatureChain* addEnergyGuidance (Cell* cell, UnitContext* context) const = 0;
};

