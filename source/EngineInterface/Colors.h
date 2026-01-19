#pragma once

#include "EngineConstants.h"

#include <stdint.h>

namespace Const
{
    uint32_t constexpr IndividualObjectColor1 = 0x2020FF;  //for device code
    uint32_t constexpr IndividualObjectColor2 = 0x8B20FF;
    uint32_t constexpr IndividualObjectColor3 = 0xFF20FF;
    uint32_t constexpr IndividualObjectColor4 = 0xFF2020;
    uint32_t constexpr IndividualObjectColor5 = 0xFF7F20;
    uint32_t constexpr IndividualObjectColor6 = 0xFFFF20;
    uint32_t constexpr IndividualObjectColor7 = 0x20FF20;

    uint32_t constexpr IndividualObjectColors[MAX_COLORS] = {  //array for convenience
        IndividualObjectColor1,
        IndividualObjectColor2,
        IndividualObjectColor3,
        IndividualObjectColor4,
        IndividualObjectColor5,
        IndividualObjectColor6,
        IndividualObjectColor7};
}

template <typename T>
using ColorVector = T[MAX_COLORS];

template <typename T>
using ColorMatrix = T[MAX_COLORS][MAX_COLORS];
