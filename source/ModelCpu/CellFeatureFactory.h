#pragma once

#include <QString>

#include "Model/Api/CellFeatureEnums.h"
#include "Model/Api/Definitions.h"
#include "Definitions.h"

class CellFeatureFactory
{
public:
	virtual ~CellFeatureFactory() = default;

	virtual CellFeatureChain* build(CellFeatureDescription const& desc, UnitContext* context) const = 0;
};

