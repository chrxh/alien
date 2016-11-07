#ifndef MODELFACTORY_H
#define MODELFACTORY_H

#include "model/decorators/constants.h"

#include <QtGlobal>
#include <QVector3D>

class AlienCell;
class AlienGrid;

class ModelFactory
{
public:
    virtual ~ModelFactory () {}
    virtual AlienCell* buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid) = 0;
    virtual AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
};

#endif // MODELFACTORY_H
