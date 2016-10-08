#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon();
    AlienCellFunctionWeapon (quint8* cellTypeData);
    AlienCellFunctionWeapon (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName () const;

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class WEAPON {
        OUT = 5,
    };
    enum class WEAPON_OUT {
        NO_TARGET,
        STRIKE_SUCCESSFUL
    };
};

#endif // ALIENCELLFUNCTIONWEAPON_H
