#pragma once

#include "EngineConstants.h"
#include "Colors.h"
#include "SimulationParametersTypes.h"

/**
 * NOTE: header is also included in kernel code
 */

struct ColorTransitionRules
{
    ColorVector<int> cellColorTransitionDuration = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    ColorVector<int> cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};

    bool operator==(ColorTransitionRules const&) const = default;
};

struct SimulationParametersZoneValues
{
    bool radiationDisableSources = false;

    ColorVector<float> maxAgeForInactiveCells = {   // Candidate for deletion
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value,
        Infinity<float>::value};
    ColorTransitionRules colorTransitionRules;

    bool operator==(SimulationParametersZoneValues const&) const = default;
};
