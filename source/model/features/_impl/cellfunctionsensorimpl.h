#ifndef CELLFUNCTIONSENSOR_H
#define CELLFUNCTIONSENSOR_H

#include "model/features/CellFunction.h"

class CellFunctionSensorImpl
	: public CellFunction
{
public:
    CellFunctionSensorImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONSENSOR_H
