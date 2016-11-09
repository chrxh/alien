#ifndef ALIENCELLFUNCTIONPROPULSION_H
#define ALIENCELLFUNCTIONPROPULSION_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionPropulsion : public AlienCellFunction
{
public:
    AlienCellFunctionPropulsion (AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::PROPULSION; }

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENCELLFUNCTIONPROPULSION_H
