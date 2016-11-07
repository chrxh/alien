#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

class AlienCell;
class AlienGrid;

class EntityFactory
{
public:
    virtual ~EntityFactory () {}
    virtual AlienCell* buildCellWithRandomData (qreal energy, AlienGrid*& grid) = 0;
    virtual AlienCell* buildCell (qreal energy, AlienGrid*& grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D()) = 0;
    virtual AlienCell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
        , AlienGrid*& grid) = 0;
    virtual AlienCell* buildCellWithoutConnectingCells (QDataStream& stream, AlienGrid*& grid) = 0;
};

#endif // ENTITYFACTORY_H
