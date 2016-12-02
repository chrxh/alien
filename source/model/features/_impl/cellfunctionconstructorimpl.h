#ifndef CELLFUNCTIONCONSTRUCTOR_H
#define CELLFUNCTIONCONSTRUCTOR_H

#include "model/features/cellfunction.h"

#include <QVector3D>

class CellFunctionConstructorImpl : public CellFunction
{
public:
    CellFunctionConstructorImpl (Grid* grid);

    CellFunctionType getType () const { return CellFunctionType::CONSTRUCTOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // CELLFUNCTIONCONSTRUCTOR_H
