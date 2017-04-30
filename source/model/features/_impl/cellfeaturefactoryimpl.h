#ifndef CELLFEATUREFACTORYIMPL_H
#define CELLFEATUREFACTORYIMPL_H

#include "model/features/cellfeaturefactory.h"

class CellFeatureFactoryImpl : public CellFeatureFactory
{
public:
    CellFeatureFactoryImpl ();
    ~CellFeatureFactoryImpl () {}

    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, SimulationUnitContext* context) const override;
    CellFeature* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data, SimulationUnitContext* context) const override;

    CellFeature* addEnergyGuidance (Cell* cell, SimulationUnitContext* context) const override;
};

#endif // CELLFEATUREFACTORYIMPL_H
