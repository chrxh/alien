#pragma once

#include "Definitions.h"

struct NeuralNetWeight
{
    NeuralNetWeight() = default;

    HOST_DEVICE NeuralNetWeight(float value)
    {
        float scaled = value * NeuralNetWeightScale;

        int intScaled;
        if (scaled >= 0.0f) {
            intScaled = static_cast<int>(scaled + 0.5f);
        } else {
            intScaled = static_cast<int>(scaled - 0.5f);
        }

        if (intScaled < -128) {
            intScaled = -128;
        }
        if (intScaled > 127) {
            intScaled = 127;
        }
        rawValue = static_cast<int8_t>(intScaled);
    }

    HOST_DEVICE static NeuralNetWeight fromRawValue(uint8_t rawValue)
    {
        NeuralNetWeight result;
        result.rawValue = rawValue;
        return result;
    }

    HOST_DEVICE float getValue() const { return static_cast<float>(rawValue) / NeuralNetWeightScale; }

    bool operator==(NeuralNetWeight const&) const = default;

    int8_t rawValue;

    static auto constexpr NeuralNetWeightScale = 32.0f;
};
