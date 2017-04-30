#ifndef CELLFUNCTIONPROPULSION_H
#define CELLFUNCTIONPROPULSION_H

#include "model/features/cellfunction.h"

class CellFunctionPropulsionImpl : public CellFunction
{
public:
    CellFunctionPropulsionImpl (SimulationUnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::PROPULSION; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
	SimulationParameters* _parameters = nullptr;
};

#endif // CELLFUNCTIONPROPULSION_H
