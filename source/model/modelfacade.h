#ifndef MODELFACADE_H
#define MODELFACADE_H

#include "model/entities/aliencellto.h"
#include "model/decorators/constants.h"

#include <QtGlobal>
#include <QVector3D>

class AlienCell;
class AlienGrid;

class ModelFacade
{
public:
    virtual ~ModelFacade () {}

    virtual AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, AlienGrid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual AlienCell* buildDecoratedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, AlienGrid*& grid) = 0;
    virtual AlienCell* buildDecoratedCell (QDataStream& stream, AlienGrid*& grid) = 0;
    virtual AlienCell* buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid) = 0;
    virtual AlienCellTO buildCellTO (AlienCell* cell) = 0;
};

#endif // MODELFACADE_H
