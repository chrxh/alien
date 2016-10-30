#ifndef ALIENCELLDECORATORFACTORYIMPL_H
#define ALIENCELLDECORATORFACTORYIMPL_H

#include "model/decorators/aliencelldecoratorfactory.h"

class AlienCellDecoratorFactoryImpl : public AlienCellDecoratorFactory
{
public:
    ~AlienCellDecoratorFactoryImpl () {}

    AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, AlienCellFunction::Type type, quint8* data, AlienGrid*& grid);
    AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, AlienCellFunction::Type type, AlienGrid*& grid);
    AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);
    AlienCellFunctionDecorator* addRandomCellFunction (AlienCell* cell, AlienGrid*& grid);

    AlienEnergyGuidanceDecorator* addEnergyGuidance (AlienCell* cell, AlienGrid*& grid);
};

#endif // ALIENCELLDECORATORFACTORYIMPL_H
