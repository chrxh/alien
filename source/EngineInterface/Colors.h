#pragma once

#include <stdint.h>

#include "EngineConstants.h"

namespace Const
{
    uint32_t constexpr IndividualCellColor1 = 0x0000FF;  //for device code
    uint32_t constexpr IndividualCellColor2 = 0x8B00FF;
    uint32_t constexpr IndividualCellColor3 = 0xFF00FF;
    uint32_t constexpr IndividualCellColor4 = 0xFF0000;
    uint32_t constexpr IndividualCellColor5 = 0xFF7F00;
    uint32_t constexpr IndividualCellColor6 = 0xFFFF00;
    uint32_t constexpr IndividualCellColor7 = 0x00FF00;

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
