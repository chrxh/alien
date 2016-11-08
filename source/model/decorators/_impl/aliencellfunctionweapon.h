#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon (AlienCell* cell, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::WEAPON; }
};

#endif // ALIENCELLFUNCTIONWEAPON_H
