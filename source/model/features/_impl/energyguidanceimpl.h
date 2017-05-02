#ifndef ENERGYGUIDANCEDECORATORIMPL_H
#define ENERGYGUIDANCEDECORATORIMPL_H

#include "model/features/EnergyGuidance.h"

class EnergyGuidanceImpl
	: public EnergyGuidance
{
public:
    EnergyGuidanceImpl (UnitContext* context);

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // ENERGYGUIDANCEDECORATORIMPL_H
