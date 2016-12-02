#ifndef CELLFUNCTIONPROPULSION_H
#define CELLFUNCTIONPROPULSION_H

#include "model/features/cellfunction.h"

class CellFunctionPropulsionImpl : public CellFunction
{
public:
    CellFunctionPropulsionImpl (Grid* grid);

    CellFunctionType getType () const { return CellFunctionType::PROPULSION; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // CELLFUNCTIONPROPULSION_H
