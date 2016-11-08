#ifndef ALIENCELLFUNCTIONPROPULSION_H
#define ALIENCELLFUNCTIONPROPULSION_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionPropulsion : public AlienCellFunction
{
public:
    AlienCellFunctionPropulsion (AlienCell* cell, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::PROPULSION; }
};

#endif // ALIENCELLFUNCTIONPROPULSION_H
