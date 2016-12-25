#ifndef CELLFUNCTIONSENSOR_H
#define CELLFUNCTIONSENSOR_H

#include "model/features/cellfunction.h"

class CellFunctionSensorImpl : public CellFunction
{
public:
    CellFunctionSensorImpl (SimulationContext* context);

    CellFunctionType getType () const { return CellFunctionType::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONSENSOR_H
