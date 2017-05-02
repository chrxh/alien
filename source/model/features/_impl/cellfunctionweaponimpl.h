#ifndef CELLFUNCTIONWEAPON_H
#define CELLFUNCTIONWEAPON_H

#include "model/features/cellfunction.h"

class CellFunctionWeaponImpl : public CellFunction
{
public:
    CellFunctionWeaponImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::WEAPON; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONWEAPON_H
