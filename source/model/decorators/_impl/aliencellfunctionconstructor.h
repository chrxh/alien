#ifndef ALIENCELLFUNCTIONCONSTRUCTOR_H
#define ALIENCELLFUNCTIONCONSTRUCTOR_H

#include "model/decorators/aliencellfunction.h"

#include <QVector3D>

class AlienCellCluster;
class AlienCellFunctionConstructor : public AlienCellFunction
{
public:
    AlienCellFunctionConstructor (AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::CONSTRUCTOR; }

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENCELLFUNCTIONCONSTRUCTOR_H
