#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    Cell* buildCell (qreal energy, Grid*& grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D());
    Cell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid*& grid);
    Cell* buildCellWithRandomData (qreal energy, Grid*& grid);
};

#endif // CELLFACTORYIMPL_H
