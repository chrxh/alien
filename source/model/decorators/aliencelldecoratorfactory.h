#ifndef ALIENTOKENPROCESSINGFACTORY_H
#define ALIENTOKENPROCESSINGFACTORY_H

#include <QString>

class AlienGrid;
class AlienCellFunction;
class AlienCellFunctionFactory
{
public:
    static AlienCellFunction* build (Type type, quint8* cellFunctionData, AlienGrid*& grid);
    static AlienCellFunction* build (Type type, AlienGrid*& grid);
    static AlienCellFunction* build (QDataStream& stream, AlienGrid*& grid);
    static AlienCellFunction* buildRandomCellFunction (AlienGrid*& grid);

    static Type getCellFunctionType ();

    enum class Type {
        COMPUTER,
        PROPULSION,
        SCANNER,
        WEAPON,
        CONSTRUCTOR,
        SENSOR,
        COMMUNICATOR,
        _COUNTER
    };
};

#endif // ALIENTOKENPROCESSINGFACTORY_H
