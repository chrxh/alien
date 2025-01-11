#pragma once

#include <stdint.h>
#include <limits>

#include "EngineConstants.h"

namespace Const
{
    uint32_t const IndividualCellColor1 = 0x5070ff; //for device code
    uint32_t const IndividualCellColor2 = 0xff6040;
    uint32_t const IndividualCellColor3 = 0x70ff50;
    uint32_t const IndividualCellColor4 = 0xffdf50;
    uint32_t const IndividualCellColor5 = 0xbf50ff;
    uint32_t const IndividualCellColor6 = 0x50ffef;
    uint32_t const IndividualCellColor7 = 0xbfbfbf;

    uint32_t const IndividualCellColors[MAX_COLORS] = {  //array for convenience
        IndividualCellColor1,
        IndividualCellColor2,
        IndividualCellColor3,
        IndividualCellColor4,
        IndividualCellColor5,
        IndividualCellColor6,
        IndividualCellColor7};

    uint32_t const CellTypeInfoColor = 0x404090;
    uint32_t const BranchNumberInfoColor = 0x000000;

    uint32_t const VoidColor = 0x000000;
}

template <typename T>
using ColorVector = T[MAX_COLORS];

template <typename T>
using ColorMatrix = T[MAX_COLORS][MAX_COLORS];

template <typename T>
struct Infinity
{
    static auto constexpr value = std::numeric_limits<T>::max();
};

