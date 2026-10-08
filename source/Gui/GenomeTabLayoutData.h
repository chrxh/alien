#pragma once

#include "Definitions.h"

struct _GenomeTabLayoutData
{
    bool initialized = false;
    float genomeEditorWidth = 0;  // Left field: genome properties, mutation rates and the gene/node tree
    float inspectorWidth = 0;     // Middle field: properties of the selected gene or node
    float desiredConfigurationPreviewWidth = 300.0f;
    float structureHeight = 0;  // Share of the left field taken by the gene/node tree
    float neuralNetEditorHeight = 0;

    GenomeTabLayoutData clone() const
    {
        auto result = std::make_shared<_GenomeTabLayoutData>();
        *result = *this;
        return result;
    }
};
