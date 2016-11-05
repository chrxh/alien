#ifndef ALIENCELLFACTORYIMPL_H
#define ALIENCELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    ~EntityFactoryImpl () {}

    AlienCell* buildCellWithRandomData (qreal energy, AlienGrid*& grid) = 0;
    AlienCell* buildCell (qreal energy, AlienGrid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0, AlienCellFunction* cellFunction = 0
        , QVector3D relPos = QVector3D()) = 0;
    AlienCell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
        , AlienGrid*& grid) = 0;
    AlienCell* buildCellWithoutConnectingCells (QDataStream& stream, AlienGrid*& grid) = 0;
};

#endif // ALIENCELLFACTORYIMPL_H
