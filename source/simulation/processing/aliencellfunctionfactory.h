#ifndef ALIENTOKENPROCESSINGFACTORY_H
#define ALIENTOKENPROCESSINGFACTORY_H

#include <QString>

class AlienCellFunction;
class AlienCellFunctionFactory
{
public:
    static AlienCellFunction* build (QString type, bool randomData);
    static AlienCellFunction* build (QDataStream& stream);
    static AlienCellFunction* build (QString type, quint8* cellTypeData);
    static AlienCellFunction* buildRandom (bool randomData);
//    static int convertFunctionNameToCellType (QString name);
};

#endif // ALIENTOKENPROCESSINGFACTORY_H
