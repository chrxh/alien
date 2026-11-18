#pragma once

#include "CellTypeConstants.h"
#include "Definitions.h"

class DistanceSignal
{
public:
    HOST_DEVICE static float encode(float distance) { return 1.0f / (1.0f + distance / Const::DistanceAtHalfSignal); }

    HOST_DEVICE static bool isInfinite(float signal) { return signal <= 0.0f; }

    HOST_DEVICE static float decode(float signal)
    {
        auto clampedSignal = signal < 1.0f ? signal : 1.0f;
        return Const::DistanceAtHalfSignal * (1.0f / clampedSignal - 1.0f);
    }

    HOST_DEVICE static bool isAtLeastAsClose(float signal, float otherSignal)
    {
        if (isInfinite(otherSignal)) {
            return true;
        }
        return !isInfinite(signal) && decode(signal) <= decode(otherSignal);
    }
};
