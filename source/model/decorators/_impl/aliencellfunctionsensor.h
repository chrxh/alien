#ifndef ALIENCELLFUNCTIONSENSOR_H
#define ALIENCELLFUNCTIONSENSOR_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionSensor : public AlienCellFunction
{
public:
    AlienCellFunctionSensor (AlienCell* cell, AlienGrid*& grid);
    AlienCellFunctionSensor (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionSensor (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::SENSOR; }
};

#endif // ALIENCELLFUNCTIONSENSOR_H
