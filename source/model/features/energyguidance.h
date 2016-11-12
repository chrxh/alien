#ifndef ENERGYGUIDANCE_H
#define ENERGYGUIDANCE_H

#include "cellfeature.h"

class EnergyGuidance: public CellFeature
{
public:
    EnergyGuidance (Grid*& grid) : CellFeature(grid) {}

    virtual ~EnergyGuidance () {}
};

#endif // ENERGYGUIDANCE_H
