#ifndef ALIENCELLFUNCTIONWEAPON_H
#define ALIENCELLFUNCTIONWEAPON_H

#include "aliencellfunction.h"

class AlienCellFunctionWeapon : public AlienCellFunction
{
public:
    AlienCellFunctionWeapon(AlienGrid*& grid);
    AlienCellFunctionWeapon (quint8* cellTypeData, AlienGrid*& grid);
    AlienCellFunctionWeapon (QDataStream& stream, AlienGrid*& grid);

    void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose);
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
