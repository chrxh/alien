#ifndef ALIENCELLDECORATORFACTORYIMPL_H
#define ALIENCELLDECORATORFACTORYIMPL_H

#include "model/decorators/aliencelldecoratorfactory.h"

class AlienCellDecoratorFactoryImpl : public AlienCellDecoratorFactory
{
public:
    ~AlienCellDecoratorFactoryImpl () {}

    AlienCellFunction* addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid);
    AlienCellFunction* addCellFunction (AlienCell* cell, CellFunctionType type, AlienGrid*& grid);
    AlienCellFunction* addCellFunction (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);
    AlienCellFunction* addRandomCellFunction (AlienCell* cell, AlienGrid*& grid);

    AlienEnergyGuidance* addEnergyGuidance (AlienCell* cell, AlienGrid*& grid);
};

#endif // ALIENCELLDECORATORFACTORYIMPL_H
