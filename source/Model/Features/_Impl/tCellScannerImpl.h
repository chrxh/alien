#ifndef CELLFUNCTIONSCANNER_H
#define CELLFUNCTIONSCANNER_H

#include "model/Features/CellFunction.h"

class CellScannerImpl
	: public CellFunction
{
public:
    CellScannerImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SCANNER; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONSCANNER_H
