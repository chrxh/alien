#pragma once

#include "Definitions.h"

// Draws the genes as boxes and their constructors as arrows, colored by the issues of the genome; a click selects the gene or the constructing node
class GeneGraphWidget
{
public:
    void process(GenomeTabEditData const& editData, float height);
};
