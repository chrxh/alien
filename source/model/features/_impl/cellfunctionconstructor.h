#ifndef CELLFUNCTIONCONSTRUCTOR_H
#define CELLFUNCTIONCONSTRUCTOR_H

#include "model/features/cellfunction.h"

#include <QVector3D>

class CellCluster;
class CellFunctionConstructor : public CellFunction
{
public:
    CellFunctionConstructor (Grid*& grid);

    CellFunctionType getType () const { return CellFunctionType::CONSTRUCTOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
};

#endif // CELLFUNCTIONCONSTRUCTOR_H
