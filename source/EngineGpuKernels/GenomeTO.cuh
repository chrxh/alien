#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"

struct NodeTO
{
    CellType type;
};

struct GeneTO
{
    int numNodes;
    NodeTO* nodes;
};

struct GenomeTO_New
{
    uint64_t id;

    int numGenes;
    GeneTO* genes;
};
