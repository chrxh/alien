#ifndef ENERGYGUIDANCE_H
#define ENERGYGUIDANCE_H

#include "cellfeature.h"

class EnergyGuidance: public CellDecorator
{
public:
    EnergyGuidance (Grid*& grid) : CellDecorator(grid) {}

    virtual ~EnergyGuidance () {}
};

#endif // ENERGYGUIDANCE_H
