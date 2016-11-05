#ifndef ALIENCELLFUNCTIONCONSTRUCTOR_H
#define ALIENCELLFUNCTIONCONSTRUCTOR_H

#include "model/decorators/aliencellfunction.h"

#include <QVector3D>

class AlienCellCluster;
class AlienCellFunctionConstructor : public AlienCellFunction
{
public:
    AlienCellFunctionConstructor (AlienCell* cell, AlienGrid*& grid);
    AlienCellFunctionConstructor (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionConstructor (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::CONSTRUCTOR; }

    void serialize (QDataStream& stream);
};

#endif // ALIENCELLFUNCTIONCONSTRUCTOR_H
