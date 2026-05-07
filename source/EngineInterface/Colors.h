#pragma once

#include <stdint.h>

#include <Base/MathTypes.h>

#include "Definitions.h"
#include "EngineConstants.h"

namespace Const
{
    uint32_t constexpr IndividualObjectColor1 = 0x2020FF;  //for device code
    uint32_t constexpr IndividualObjectColor2 = 0xB520FF;
    uint32_t constexpr IndividualObjectColor3 = 0xFF20B5;
    uint32_t constexpr IndividualObjectColor4 = 0xFF2020;
    uint32_t constexpr IndividualObjectColor5 = 0xFF9020;
    uint32_t constexpr IndividualObjectColor6 = 0xFFFF20;
    uint32_t constexpr IndividualObjectColor7 = 0x90FF20;
    uint32_t constexpr IndividualObjectColor8 = 0x20FF20;
    uint32_t constexpr IndividualObjectColor9 = 0x20FFB5;
    uint32_t constexpr IndividualObjectColor10 = 0xFFFFFF;

    uint32_t constexpr DefaultIndividualObjectColors[MAX_COLORS] = {  //array for convenience
        IndividualObjectColor1,
        IndividualObjectColor2,
        IndividualObjectColor3,
        IndividualObjectColor4,
        IndividualObjectColor5,
        IndividualObjectColor6,
        IndividualObjectColor7,
        IndividualObjectColor8,
        IndividualObjectColor9,
        IndividualObjectColor10};

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

constexpr ColorVector<FloatColorRGB> createDefaultIndividualObjectColorVector()
{
    ColorVector<FloatColorRGB> result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = toFloatColorRGB(Const::DefaultIndividualObjectColors[i]);
    }
    return result;
}

inline uint32_t toRgbColor(FloatColorRGB const& color)
{
    auto const toInt = [](float value) {
        if (value < 0.0f) {
            value = 0.0f;
        } else if (value > 1.0f) {
            value = 1.0f;
        }
        return static_cast<uint32_t>(value * 255.0f + 0.5f);
    };
    return (toInt(color.r) << 16) | (toInt(color.g) << 8) | toInt(color.b);
}

inline uint32_t getIndividualObjectColor(ColorVector<FloatColorRGB> const& colors, int color)
{
    return toRgbColor(colors.values[color]);
}
