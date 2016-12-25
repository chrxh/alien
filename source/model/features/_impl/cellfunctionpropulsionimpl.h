#ifndef CELLFUNCTIONPROPULSION_H
#define CELLFUNCTIONPROPULSION_H

#include "model/features/cellfunction.h"

class CellFunctionPropulsionImpl : public CellFunction
{
public:
    CellFunctionPropulsionImpl (SimulationContext* context);

    CellFunctionType getType () const { return CellFunctionType::PROPULSION; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONPROPULSION_H
