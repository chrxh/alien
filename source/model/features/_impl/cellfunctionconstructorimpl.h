#ifndef CELLFUNCTIONCONSTRUCTOR_H
#define CELLFUNCTIONCONSTRUCTOR_H

#include "model/features/CellFunction.h"

#include <QVector3D>

class CellFunctionConstructorImpl
	: public CellFunction
{
public:
    CellFunctionConstructorImpl (UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::CONSTRUCTOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
};

#endif // CELLFUNCTIONCONSTRUCTOR_H
