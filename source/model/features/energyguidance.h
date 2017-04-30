#ifndef ENERGYGUIDANCE_H
#define ENERGYGUIDANCE_H

#include "cellfeature.h"

class EnergyGuidance: public CellFeature
{
public:
    EnergyGuidance (SimulationUnitContext* context) : CellFeature(context) {}

    virtual ~EnergyGuidance () {}
};

#endif // ENERGYGUIDANCE_H
