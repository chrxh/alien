#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

class Cell;
class Grid;

class EntityFactory
{
public:
    virtual ~EntityFactory () {}
    virtual Cell* buildCell (qreal energy, Grid*& grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid*& grid) = 0;
    virtual Cell* buildCellWithRandomData (qreal energy, Grid*& grid) = 0;
};

#endif // ENTITYFACTORY_H
