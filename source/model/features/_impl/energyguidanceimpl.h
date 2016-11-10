#ifndef ENERGYGUIDANCEDECORATORIMPL_H
#define ENERGYGUIDANCEDECORATORIMPL_H

#include "model/features/energyguidance.h"

class EnergyGuidanceImpl : public EnergyGuidance
{
public:
    EnergyGuidanceImpl (Grid*& grid);

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // ENERGYGUIDANCEDECORATORIMPL_H
