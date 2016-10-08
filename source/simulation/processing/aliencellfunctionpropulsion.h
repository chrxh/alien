#ifndef ALIENTOKENPROCESSINGPROPULSION_H
#define ALIENTOKENPROCESSINGPROPULSION_H

#include "aliencellfunction.h"

class AlienCellFunctionPropulsion : public AlienCellFunction
{
public:
    AlienCellFunctionPropulsion ();
    AlienCellFunctionPropulsion (quint8* cellTypeData);
    AlienCellFunctionPropulsion (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName () const;

    //constants for cell function programming
    enum class PROP {
        OUT = 5,
        IN = 8,
        IN_ANGLE = 9,
        IN_POWER = 10
    };
    enum class PROP_OUT {
        SUCCESS,
        SUCCESS_DAMPING_FINISHED,
        ERROR_NO_ENERGY
    };
    enum class PROP_IN {
        DO_NOTHING,
        BY_ANGLE,
        FROM_CENTER,
        TOWARD_CENTER,
        ROTATION_CLOCKWISE,
        ROTATION_COUNTERCLOCKWISE,
        DAMP_ROTATION
    };

private:
    qreal convertDataToThrustPower (quint8 b);
};

#endif // ALIENTOKENPROCESSINGPROPULSION_H
