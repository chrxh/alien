#pragma once

#include "EngineConstants.h"
#include "Colors.h"
#include "SimulationParametersTypes.h"

/**
 * NOTE: header is also included in kernel code
 */

struct SimulationParametersZoneValues
{
    bool radiationDisableSources = false;

    bool operator==(SimulationParametersZoneValues const&) const = default;
};
