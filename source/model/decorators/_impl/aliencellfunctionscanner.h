#ifndef ALIENCELLFUNCTIONSCANNER_H
#define ALIENCELLFUNCTIONSCANNER_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionScanner : public AlienCellFunction
{
public:
    AlienCellFunctionScanner (AlienCell* cell, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::SCANNER; }
};

#endif // ALIENCELLFUNCTIONSCANNER_H
