#ifndef ALIENTOKENPROCESSINGFACTORY_H
#define ALIENTOKENPROCESSINGFACTORY_H

#include <QString>

class AlienGrid;
class AlienCellFunction;
class AlienCellFunctionFactory
{
public:
    static AlienCellFunction* build (QString name, quint8* cellFunctionData, AlienGrid*& grid);
    static AlienCellFunction* build (QString name, AlienGrid*& grid);
    static AlienCellFunction* build (QDataStream& stream, AlienGrid*& grid);
    static AlienCellFunction* buildRandomCellFunction (AlienGrid*& grid);
};

#endif // ALIENTOKENPROCESSINGFACTORY_H
