#ifndef MODELFACTORYIMPL_H
#define MODELFACTORYIMPL_H

#include "model/modelfacade.h"

class ModelFacadeImpl : public ModelFacade
{
public:
    ModelFacadeImpl ();
    ~ModelFacadeImpl () {}

    Cell* buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, Grid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildDecoratedCell (qreal energy, CellFunctionType type, Grid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildDecoratedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid*& grid);
    Cell* buildDecoratedCell (QDataStream& stream, Grid*& grid);
    Cell* buildDecoratedCellWithRandomData (qreal energy, Grid*& grid);
    CellTO buildCellTO (Cell* cell);
};

#endif // MODELFACTORYIMPL_H
