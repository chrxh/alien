#pragma once

#include "Model/Local/CellFunction.h"

#include <QVector2D>

class ConstructorFunction
	: public CellFunction
{
public:
    ConstructorFunction (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::CONSTRUCTOR; }

protected:
	ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;

private:
};
