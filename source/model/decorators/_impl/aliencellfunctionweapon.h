#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon (AlienCell* cell, AlienGrid*& grid);
    AlienCellFunctionWeapon (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionWeapon (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::WEAPON; }
};

#endif // ALIENCELLFUNCTIONWEAPON_H
