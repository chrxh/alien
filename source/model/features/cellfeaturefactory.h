#ifndef CELLDECORATORFACTORY_H
#define CELLDECORATORFACTORY_H

#include <QString>

#include "cellfeatureconstants.h"
#include "model/definitions.h"

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, SimulationContext* context) const = 0;
    virtual CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, SimulationContext* context) const = 0;

    virtual CellFeature* addEnergyGuidance (Cell* cell, SimulationContext* context) const = 0;
};

#endif // CELLDECORATORFACTORY_H
