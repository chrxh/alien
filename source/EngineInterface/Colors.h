#pragma once

#include "EngineConstants.h"

#include <stdint.h>

namespace Const
{
    uint32_t constexpr IndividualCellColor1 = 0x2020FF;  //for device code
    uint32_t constexpr IndividualCellColor2 = 0x8B20FF;
    uint32_t constexpr IndividualCellColor3 = 0xFF20FF;
    uint32_t constexpr IndividualCellColor4 = 0xFF2020;
    uint32_t constexpr IndividualCellColor5 = 0xFF7F20;
    uint32_t constexpr IndividualCellColor6 = 0xFFFF20;
    uint32_t constexpr IndividualCellColor7 = 0x20FF20;

    uint32_t constexpr IndividualCellColors[MAX_COLORS] = {  //array for convenience
        IndividualCellColor1,
        IndividualCellColor2,
        IndividualCellColor3,
        IndividualCellColor4,
        IndividualCellColor5,
        IndividualCellColor6,
        IndividualCellColor7};
}

template <typename T>
using ColorVector = T[MAX_COLORS];

template <typename T>
using ColorMatrix = T[MAX_COLORS][MAX_COLORS];
