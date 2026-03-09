#pragma once

#include "EngineConstants.h"

#include <stdint.h>

namespace Const
{
    uint32_t constexpr IndividualObjectColor1 = 0x2020FF;   //for device code
    uint32_t constexpr IndividualObjectColor2 = 0xB520FF;
    uint32_t constexpr IndividualObjectColor3 = 0xFF20B5;
    uint32_t constexpr IndividualObjectColor4 = 0xFF2020;
    uint32_t constexpr IndividualObjectColor5 = 0xFF9020;
    uint32_t constexpr IndividualObjectColor6 = 0xFFFF20;
    uint32_t constexpr IndividualObjectColor7 = 0x90FF20;
    uint32_t constexpr IndividualObjectColor8 = 0x20FF20;
    uint32_t constexpr IndividualObjectColor9 = 0x20FFB5;
    uint32_t constexpr IndividualObjectColor10 = 0x20C7FF;

    uint32_t constexpr IndividualObjectColors[MAX_COLORS] = {  //array for convenience
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
using ColorVector = T[MAX_COLORS];

template <typename T>
using ColorMatrix = T[MAX_COLORS][MAX_COLORS];
