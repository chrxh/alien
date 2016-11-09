#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon (AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::WEAPON; }

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENCELLFUNCTIONWEAPON_H
