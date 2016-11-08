#ifndef ALIENCELLFUNCTIONCONSTRUCTOR_H
#define ALIENCELLFUNCTIONCONSTRUCTOR_H

#include "model/decorators/aliencellfunction.h"

#include <QVector3D>

class AlienCellCluster;
class AlienCellFunctionConstructor : public AlienCellFunction
{
public:
    AlienCellFunctionConstructor (AlienCell* cell, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::CONSTRUCTOR; }
};

#endif // ALIENCELLFUNCTIONCONSTRUCTOR_H
