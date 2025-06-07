#pragma once

#include "Definitions.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void processGeneList();

    CreatureTabGenomeData _genome;
    std::optional<int> _selectedGene;

    // Layout data
    CreatureTabLayoutData _layoutData;
};
