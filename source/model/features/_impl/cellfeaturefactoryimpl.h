#ifndef CELLFEATUREFACTORYIMPL_H
#define CELLFEATUREFACTORYIMPL_H

#include "model/features/cellfeaturefactory.h"

class CellDecoratorFactoryImpl : public CellFeatureFactory
{
public:
    CellDecoratorFactoryImpl ();
    ~CellDecoratorFactoryImpl () {}

    void addCellFunction (Cell* cell, CellFunctionType type, SimulationContext* context) const override;
    void addCellFunction (Cell* cell, CellFunctionType type, quint8* data, SimulationContext* context) const override;
    void addCellFunction (Cell* cell, CellFunctionType type, QDataStream& stream, SimulationContext* context) const override;

    void addEnergyGuidance (Cell* cell, SimulationContext* context) const override;
};

#endif // CELLFEATUREFACTORYIMPL_H
