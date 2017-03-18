#ifndef CELLFUNCTIONWEAPON_H
#define CELLFUNCTIONWEAPON_H

#include "model/features/cellfunction.h"

class CellFunctionWeaponImpl : public CellFunction
{
public:
    CellFunctionWeaponImpl (SimulationContext* context);

    CellFunctionType getType () const { return CellFunctionType::WEAPON; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
    CellMap* _cellMap = nullptr;
	SimulationParameters* _parameters = nullptr;
};

#endif // CELLFUNCTIONWEAPON_H
