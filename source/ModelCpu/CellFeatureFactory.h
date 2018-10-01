#pragma once

#include <QString>

#include "ModelInterface/CellFeatureEnums.h"
#include "ModelInterface/Definitions.h"
#include "Definitions.h"

class CellFeatureFactory
{
public:
	virtual ~CellFeatureFactory() = default;

	virtual CellFeatureChain* build(CellFeatureDescription const& desc, UnitContext* context) const = 0;
};

