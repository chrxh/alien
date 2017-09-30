#pragma once

#include <QString>

#include "Model/Api/Definitions.h"
#include "Model/Api/CellFeatureEnums.h"

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, UnitContext* context) const = 0;
    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const = 0;

    virtual CellFeature* addEnergyGuidance (Cell* cell, UnitContext* context) const = 0;
};

