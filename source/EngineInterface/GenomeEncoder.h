#pragma once

#include <vector>

#include "EngineInterface/Descriptions.h"

class GenomeEncoder
{
public:
    static std::vector<uint8_t> encode(std::vector<CellDescription> const& cells, float initialAngle = 0);
};
