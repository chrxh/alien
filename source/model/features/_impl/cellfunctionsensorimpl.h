#ifndef CELLFUNCTIONSENSOR_H
#define CELLFUNCTIONSENSOR_H

#include "model/features/cellfunction.h"

class CellFunctionSensorImpl : public CellFunction
{
public:
    CellFunctionSensorImpl (SimulationContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SENSOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
    CellMap* _cellMap = nullptr;
    Topology* _topology = nullptr;
	SimulationParameters* _parameters = nullptr;
};

#endif // CELLFUNCTIONSENSOR_H
