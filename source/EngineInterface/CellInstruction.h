#pragma once

#include <cstdint>

#include "Enums.h"

struct CellInstruction
{
    Enums::ComputationOperation operation;
    Enums::ComputationOpType opType1;
    Enums::ComputationOpType opType2;
    uint8_t operand1;
    uint8_t operand2;
};
