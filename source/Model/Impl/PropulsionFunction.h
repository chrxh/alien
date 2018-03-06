#pragma once

#include "Model/Local/CellFunction.h"

class PropulsionFunction
	: public CellFunction
{
public:
    PropulsionFunction (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::PROPULSION; }

protected:
	ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;
};
