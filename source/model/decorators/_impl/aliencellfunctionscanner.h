#ifndef ALIENCELLFUNCTIONSCANNER_H
#define ALIENCELLFUNCTIONSCANNER_H

#include "model/decorators/aliencellfunction.h"

class AlienCellFunctionScanner : public AlienCellFunction
{
public:
    AlienCellFunctionScanner (AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::SCANNER; }

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENCELLFUNCTIONSCANNER_H
