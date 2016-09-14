#ifndef ALIENCELLFUNCTIONSENSOR_H
#define ALIENCELLFUNCTIONSENSOR_H

#include "aliencellfunction.h"

class AlienCellFunctionSensor : public AlienCellFunction
{
public:
    AlienCellFunctionSensor ();
    AlienCellFunctionSensor (quint8* cellTypeData);
    AlienCellFunctionSensor (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);
};

#endif // ALIENCELLFUNCTIONSENSOR_H
