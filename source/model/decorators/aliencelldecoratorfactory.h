#ifndef ALIENCELLDECORATORFACTORY_H
#define ALIENCELLDECORATORFACTORY_H

#include "constants.h"

#include <QString>

class AlienGrid;
class AlienCell;
class AlienCellFunction;
class AlienEnergyGuidance;

class AlienCellDecoratorFactory
{
public:
    virtual ~AlienCellDecoratorFactory () {}

    virtual void addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid) = 0;
    virtual void addCellFunction (AlienCell* cell, CellFunctionType type, AlienGrid*& grid) = 0;
    virtual void addCellFunction (AlienCell* cell, CellFunctionType type, QDataStream& stream, AlienGrid*& grid) = 0;

    virtual void addEnergyGuidance (AlienCell* cell, AlienGrid*& grid) = 0;
};

#endif // ALIENCELLDECORATORFACTORY_H
