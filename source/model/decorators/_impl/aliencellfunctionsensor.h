#ifndef ALIENCELLFUNCTIONSENSOR_H
#define ALIENCELLFUNCTIONSENSOR_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionSensor : public AlienCellFunction
{
public:
    AlienCellFunctionSensor (AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::SENSOR; }

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENCELLFUNCTIONSENSOR_H
