#ifndef CELLFEATUREFACTORYIMPL_H
#define CELLFEATUREFACTORYIMPL_H

#include "model/features/cellfeaturefactory.h"

class CellDecoratorFactoryImpl : public CellFeatureFactory
{
public:
    CellDecoratorFactoryImpl ();
    ~CellDecoratorFactoryImpl () {}

    CellFeature* addCellFunction (Cell* cell, CellFunctionType type, SimulationContext* context) const override;
    CellFeature* addCellFunction (Cell* cell, CellFunctionType type, quint8* data, SimulationContext* context) const override;

    CellFeature* addEnergyGuidance (Cell* cell, SimulationContext* context) const override;
};

#endif // CELLFEATUREFACTORYIMPL_H
