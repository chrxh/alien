#pragma once

#include "CellFunction.h"

class SensorFunction
	: public CellFunction
{
public:
    SensorFunction (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

