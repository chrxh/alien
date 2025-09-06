#pragma once

#include "Definitions.h"

class _NodeEditorWidget
{
public:
    static NodeEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _NodeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void processNodeAttributes();
    void processNoSelection();

    void processNeuralNetEditor();

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    NeuralNetEditorWidget _neuralNetWidget;
};
