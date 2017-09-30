#pragma once

#include "Model/Local/CellFunction.h"

class CellWeaponImpl
	: public CellFunction
{
public:
    CellWeaponImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::WEAPON; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

