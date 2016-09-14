#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon();
    AlienCellFunctionWeapon (quint8* cellTypeData);
    AlienCellFunctionWeapon (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);
};

#endif // ALIENCELLFUNCTIONWEAPON_H
