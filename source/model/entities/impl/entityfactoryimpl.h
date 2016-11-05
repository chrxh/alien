#ifndef ALIENCELLFACTORYIMPL_H
#define ALIENCELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    AlienCell* buildCellWithRandomData (qreal energy, AlienGrid*& grid);
    AlienCell* buildCell (qreal energy, AlienGrid*& grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D());
    AlienCell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, AlienGrid*& grid);
    AlienCell* buildCellWithoutConnectingCells (QDataStream& stream, AlienGrid*& grid);
};

#endif // ALIENCELLFACTORYIMPL_H
