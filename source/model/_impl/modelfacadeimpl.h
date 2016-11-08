#ifndef MODELFACTORYIMPL_H
#define MODELFACTORYIMPL_H

#include "model/modelfacade.h"

class ModelFacadeImpl : public ModelFacade
{
public:
    ModelFacadeImpl ();
    ~ModelFacadeImpl () {}

    AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, AlienGrid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    AlienCell* buildDecoratedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, AlienGrid*& grid);
    AlienCell* buildDecoratedCell (QDataStream& stream, AlienGrid*& grid);
    AlienCell* buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid);
    AlienCellTO buildCellTO (AlienCell* cell);
};

#endif // MODELFACTORYIMPL_H
