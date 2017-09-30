#ifndef CELLFUNCTIONPROPULSION_H
#define CELLFUNCTIONPROPULSION_H

#include "Model/Local/CellFunction.h"

class CellPropulsionImpl
	: public CellFunction
{
public:
    CellPropulsionImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::PROPULSION; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;
};

#endif // CELLFUNCTIONPROPULSION_H
