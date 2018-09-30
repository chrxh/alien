#pragma once

#include "Model/Local/CellFeatureFactory.h"

class CellFeatureFactoryImpl
	: public CellFeatureFactory
{
public:
	virtual ~CellFeatureFactoryImpl() = default;

	virtual CellFeatureChain* build(CellFeatureDescription const& desc, UnitContext* context) const override;

/*
private:
    CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const;
    CellFeatureChain* addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray const& constData, QByteArray const& volatileData, UnitContext* context) const;

    CellFeatureChain* addEnergyGuidance (Cell* cell, UnitContext* context) const;
*/
};
