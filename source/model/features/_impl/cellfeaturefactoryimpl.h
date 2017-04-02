#ifndef CELLFEATUREFACTORYIMPL_H
#define CELLFEATUREFACTORYIMPL_H

#include "model/features/cellfeaturefactory.h"

class CellFeatureFactoryImpl : public CellFeatureFactory
{
public:
    CellFeatureFactoryImpl ();
    ~CellFeatureFactoryImpl () {}

    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, SimulationContext* context) const override;
    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, SimulationContext* context) const override;

    CellFeature* addEnergyGuidance (Cell* cell, SimulationContext* context) const override;
};

#endif // CELLFEATUREFACTORYIMPL_H
