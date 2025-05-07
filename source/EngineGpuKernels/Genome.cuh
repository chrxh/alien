#pragma once

#include <cstdint>

#include "EngineInterface/CellTypeConstants.h"

struct Node
{
    CellType type;
};

struct Gene
{
    int numNodes;
    Node* nodes;
};

struct Genome_New
{
    uint64_t id;

    int numGenes;
    Gene* genes;
};
