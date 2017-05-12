#ifndef CELLFUNCTIONSENSOR_H
#define CELLFUNCTIONSENSOR_H

#include "model/Features/CellFunction.h"

class CellSensorImpl
	: public CellFunction
{
public:
    CellSensorImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONSENSOR_H
