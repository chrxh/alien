#pragma once

#include "EnergyGuidance.h"

class EnergyGuidanceImpl
	: public EnergyGuidance
{
public:
    EnergyGuidanceImpl (UnitContext* context);

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};
