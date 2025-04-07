#pragma once

//NOTE: header is also included in CUDA code

struct SimulationParametersZoneEnabledValues
{
    bool radiationDisableSources = false;

    bool operator==(SimulationParametersZoneEnabledValues const&) const = default;
};
