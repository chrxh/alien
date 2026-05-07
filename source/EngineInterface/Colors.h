#pragma once

#include <stdint.h>

#include <Base/MathTypes.h>

#include "Definitions.h"
#include "EngineConstants.h"

namespace Const
{
    uint32_t constexpr CustomizationColor1 = 0x2020FF;  //for device code
    uint32_t constexpr CustomizationColor2 = 0xB520FF;
    uint32_t constexpr CustomizationColor3 = 0xFF20B5;
    uint32_t constexpr CustomizationColor4 = 0xFF2020;
    uint32_t constexpr CustomizationColor5 = 0xFF9020;
    uint32_t constexpr CustomizationColor6 = 0xFFFF20;
    uint32_t constexpr CustomizationColor7 = 0x90FF20;
    uint32_t constexpr CustomizationColor8 = 0x20FF20;
    uint32_t constexpr CustomizationColor9 = 0x20FFB5;
    uint32_t constexpr CustomizationColor10 = 0xFFFFFF;

    uint32_t constexpr DefaultCustomizationColors[MAX_COLORS] = {  //array for convenience
        CustomizationColor1,
        CustomizationColor2,
        CustomizationColor3,
        CustomizationColor4,
        CustomizationColor5,
        CustomizationColor6,
        CustomizationColor7,
        CustomizationColor8,
        CustomizationColor9,
        CustomizationColor10};

}

template <typename T>
struct ColorVector
{
    T values[MAX_COLORS] = {};

    static constexpr ColorVector uniform(T v)
    {
        ColorVector result;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = v;
        }
        return result;
    }

    HOST_DEVICE T& operator[](int i) { return values[i]; }
    HOST_DEVICE T const& operator[](int i) const { return values[i]; }

    bool operator==(ColorVector const&) const = default;
};

template <typename T>
struct ColorMatrix
{
    T values[MAX_COLORS][MAX_COLORS] = {};

    static constexpr ColorMatrix uniform(T v)
    {
        ColorMatrix result;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result.values[i][j] = v;
            }
        }
        return result;
    }

    HOST_DEVICE T* operator[](int i) { return values[i]; }
    HOST_DEVICE T const* operator[](int i) const { return values[i]; }

    bool operator==(ColorMatrix const&) const = default;
};

constexpr FloatColorRGB toFloatColorRGB(uint32_t rgb)
{
    return {static_cast<float>((rgb >> 16) & 0xff) / 255.0f, static_cast<float>((rgb >> 8) & 0xff) / 255.0f, static_cast<float>(rgb & 0xff) / 255.0f};
}

constexpr ColorVector<FloatColorRGB> getDefaultCustomizationColorVector()
{
    ColorVector<FloatColorRGB> result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = toFloatColorRGB(Const::DefaultCustomizationColors[i]);
    }
    return result;
}
