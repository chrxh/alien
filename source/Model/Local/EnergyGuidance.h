#pragma once

#include "CellFeatureChain.h"

class EnergyGuidance
	: public CellFeatureChain
{
public:
    EnergyGuidance (UnitContext* context) : CellFeatureChain(context) {}

    virtual ~EnergyGuidance () {}
};
