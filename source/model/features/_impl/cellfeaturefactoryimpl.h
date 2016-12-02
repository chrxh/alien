#ifndef CELLFEATUREFACTORYIMPL_H
#define CELLFEATUREFACTORYIMPL_H

#include "model/features/cellfeaturefactory.h"

class CellDecoratorFactoryImpl : public CellFeatureFactory
{
public:
    CellDecoratorFactoryImpl ();
    ~CellDecoratorFactoryImpl () {}

    void addCellFunction (Cell* cell, CellFunctionType type, Grid* grid);
    void addCellFunction (Cell* cell, CellFunctionType type, quint8* data, Grid* grid);
    void addCellFunction (Cell* cell, CellFunctionType type, QDataStream& stream, Grid* grid);

    void addEnergyGuidance (Cell* cell, Grid* grid);
};

#endif // CELLFEATUREFACTORYIMPL_H
