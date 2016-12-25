#ifndef CELLDECORATORFACTORY_H
#define CELLDECORATORFACTORY_H

#include <QString>

#include "cellfeatureconstants.h"
#include "model/definitions.h"

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual CellFeature* addCellFunction (Cell* cell, CellFunctionType type, quint8* data, SimulationContext* context) const = 0;
    virtual CellFeature* addCellFunction (Cell* cell, CellFunctionType type, SimulationContext* context) const = 0;

    virtual CellFeature* addEnergyGuidance (Cell* cell, SimulationContext* context) const = 0;
};

#endif // CELLDECORATORFACTORY_H
