#ifndef ALIENCELLDECORATORFACTORY_H
#define ALIENCELLDECORATORFACTORY_H

#include "aliencellfunction.h"

#include <QString>

class AlienGrid;
class AlienCell;
class AlienEnergyGuidance;

class AlienCellDecoratorFactory
{
public:
    virtual ~AlienCellDecoratorFactory () {}

    virtual AlienCellFunction* addCellFunction (AlienCell* cell, AlienCellFunction::Type type, quint8* data, AlienGrid*& grid) = 0;
    virtual AlienCellFunction* addCellFunction (AlienCell* cell, AlienCellFunction::Type type, AlienGrid*& grid) = 0;
    virtual AlienCellFunction* addCellFunction (AlienCell* cell, QDataStream& stream, AlienGrid*& grid) = 0;
    virtual AlienCellFunction* addRandomCellFunction (AlienCell* cell, AlienGrid*& grid) = 0;

    virtual AlienEnergyGuidance* addEnergyGuidance (AlienCell* cell, AlienGrid*& grid) = 0;
};

#endif // ALIENCELLDECORATORFACTORY_H
