#pragma once

#include "EngineConstants.h"

#include <stdint.h>

namespace Const
{
    uint32_t constexpr IndividualObjectColor1 = 0x2020FF;   //for device code
    uint32_t constexpr IndividualObjectColor2 = 0x7420FF;
    uint32_t constexpr IndividualObjectColor3 = 0xC720FF;
    uint32_t constexpr IndividualObjectColor4 = 0xFF20E3;
    uint32_t constexpr IndividualObjectColor5 = 0xFF2090;
    uint32_t constexpr IndividualObjectColor6 = 0xFF203C;
    uint32_t constexpr IndividualObjectColor7 = 0xFF5820;
    uint32_t constexpr IndividualObjectColor8 = 0xFFAB20;
    uint32_t constexpr IndividualObjectColor9 = 0xFFFF20;
    uint32_t constexpr IndividualObjectColor10 = 0xABFF20;
    uint32_t constexpr IndividualObjectColor11 = 0x58FF20;
    uint32_t constexpr IndividualObjectColor12 = 0x20FF3C;
    uint32_t constexpr IndividualObjectColor13 = 0x20FF90;
    uint32_t constexpr IndividualObjectColor14 = 0x20FFE3;
    uint32_t constexpr IndividualObjectColor15 = 0x20C7FF;
    uint32_t constexpr IndividualObjectColor16 = 0x2074FF;

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
        IndividualObjectColor10,
        IndividualObjectColor11,
        IndividualObjectColor12,
        IndividualObjectColor13,
        IndividualObjectColor14,
        IndividualObjectColor15,
        IndividualObjectColor16};
}

template <typename T>
using ColorVector = T[MAX_COLORS];

template <typename T>
using ColorMatrix = T[MAX_COLORS][MAX_COLORS];
