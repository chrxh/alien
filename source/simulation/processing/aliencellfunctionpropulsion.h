#ifndef ALIENTOKENPROCESSINGPROPULSION_H
#define ALIENTOKENPROCESSINGPROPULSION_H

#include "aliencellfunction.h"

class AlienCellFunctionPropulsion : public AlienCellFunction
{
public:
    AlienCellFunctionPropulsion ();
    AlienCellFunctionPropulsion (quint8* cellTypeData);
    AlienCellFunctionPropulsion (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);

private:
    qreal convertDataToThrustPower (quint8 b);
};

#endif // ALIENTOKENPROCESSINGPROPULSION_H
