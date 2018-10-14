#pragma once

#include "CellFunction.h"

class ScannerFunction
	: public CellFunction
{
public:
    ScannerFunction (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::SCANNER; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};
