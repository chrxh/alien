#ifndef ALIENTOKENPROCESSINGFACTORY_H
#define ALIENTOKENPROCESSINGFACTORY_H

#include <QString>

class AlienGrid;
class AlienCellFunction;
class AlienCellFunctionFactory
{
public:
    static AlienCellFunction* build (QString type, bool randomData, AlienGrid*& grid);
    static AlienCellFunction* build (QDataStream& stream, AlienGrid*& grid);
    static AlienCellFunction* build (QString type, quint8* cellFunctionData, AlienGrid*& grid);
    static AlienCellFunction* buildRandom (bool randomData, AlienGrid*& grid);
//    static int convertFunctionNameToCellType (QString name);
};

#endif // ALIENTOKENPROCESSINGFACTORY_H
