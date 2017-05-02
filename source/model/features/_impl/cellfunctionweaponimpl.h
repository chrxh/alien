#ifndef CELLFUNCTIONWEAPONIMPL_H
#define CELLFUNCTIONWEAPONIMPL_H

#include "model/features/CellFunction.h"

class CellFunctionWeaponImpl
	: public CellFunction
{
public:
    CellFunctionWeaponImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::WEAPON; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONWEAPONIMPL_H
