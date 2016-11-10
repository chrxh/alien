#ifndef CELLFUNCTIONPROPULSION_H
#define CELLFUNCTIONPROPULSION_H

#include "model/features/cellfunction.h"

class CellFunctionPropulsion : public CellFunction
{
public:
    CellFunctionPropulsion (Grid*& grid);

    CellFunctionType getType () const { return CellFunctionType::PROPULSION; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // CELLFUNCTIONPROPULSION_H
