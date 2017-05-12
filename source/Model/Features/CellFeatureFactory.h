#ifndef CELLFEATUREFACTORY_H
#define CELLFEATUREFACTORY_H

#include <QString>

#include "model/Definitions.h"
#include "CellFeatureEnums.h"

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, UnitContext* context) const = 0;
    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const = 0;

    virtual CellFeature* addEnergyGuidance (Cell* cell, UnitContext* context) const = 0;
};

#endif // CELLFEATUREFACTORY_H
