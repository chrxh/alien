#pragma once

#include "CellFeatureFactory.h"

class CellFeatureFactoryImpl
	: public CellFeatureFactory
{
public:
	virtual ~CellFeatureFactoryImpl() = default;

	virtual CellFeatureChain* build(CellFeatureDescription const& desc, UnitContext* context) const override;
};
