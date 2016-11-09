#ifndef ALIENCELLDECORATORFACTORYIMPL_H
#define ALIENCELLDECORATORFACTORYIMPL_H

#include "model/decorators/aliencelldecoratorfactory.h"

class AlienCellDecoratorFactoryImpl : public AlienCellDecoratorFactory
{
public:
    AlienCellDecoratorFactoryImpl ();
    ~AlienCellDecoratorFactoryImpl () {}

    void addCellFunction (AlienCell* cell, CellFunctionType type, AlienGrid*& grid);
    void addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid);
    void addCellFunction (AlienCell* cell, CellFunctionType type, QDataStream& stream, AlienGrid*& grid);

    void addEnergyGuidance (AlienCell* cell, AlienGrid*& grid);
};

#endif // ALIENCELLDECORATORFACTORYIMPL_H
