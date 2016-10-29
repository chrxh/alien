#ifndef ALIENCELLDECORATORFACTORY_H
#define ALIENCELLDECORATORFACTORY_H

#include <QString>

class AlienGrid;
class AlienCell;
class AlienCellFunctionDecorator;
class AlienEnergyGuidanceDecorator;

class AlienCellDecoratorFactory
{
public:
    virtual ~AlienCellDecoratorFactory () {}

    virtual AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, Type type, quint8* data, AlienGrid*& grid) = 0;
    virtual AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, Type type, AlienGrid*& grid) = 0;
    virtual AlienCellFunctionDecorator* addCellFunction (AlienCell* cell, QDataStream& stream, AlienGrid*& grid) = 0;
    virtual AlienCellFunctionDecorator* addRandomCellFunction (AlienCell* cell, AlienGrid*& grid) = 0;

    virtual AlienEnergyGuidanceDecorator* addEnergyGuidance (AlienCell* cell, AlienGrid*& grid) = 0;

    enum class Type {
        COMPUTER,
        PROPULSION,
        SCANNER,
        WEAPON,
        CONSTRUCTOR,
        SENSOR,
        COMMUNICATOR,
        _COUNT
    };
};

#endif // ALIENCELLDECORATORFACTORY_H
