#pragma once

#include <QString>

#include "ModelBasic/CellFeatureEnums.h"
#include "ModelBasic/Definitions.h"
#include "Definitions.h"

class CellFeatureFactory
{
public:
	virtual ~CellFeatureFactory() = default;

	virtual CellFeatureChain* build(CellFeatureDescription const& desc, UnitContext* context) const = 0;
};

