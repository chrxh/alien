#ifndef MODELFACTORYIMPL_H
#define MODELFACTORYIMPL_H

#include "model/modelfactory.h"

class ModelFactoryImpl : public ModelFactory
{
public:
    ModelFactoryImpl ();
    ~ModelFactory () {}

    AlienCell* buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid);
    AlienCell* buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid
        , int maxConnections = 0, int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());

};

#endif // MODELFACTORYIMPL_H
