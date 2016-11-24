#ifndef CELLFUNCTIONSENSOR_H
#define CELLFUNCTIONSENSOR_H

#include "model/features/cellfunction.h"

class CellFunctionSensor : public CellFunction
{
public:
    CellFunctionSensor (Grid* grid);

    CellFunctionType getType () const { return CellFunctionType::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // CELLFUNCTIONSENSOR_H
