#pragma once

#include "Definitions.h"

class _NodeEditorWidget
{
public:
    static NodeEditorWidget create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _NodeEditorWidget(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData);

    void processNodeAttributes();
    void processNoSelection();

    void processNeuralNetEditor();

    CreatureTabEditData _editData;
    CreatureTabLayoutData _layoutData;

    NeuralNetEditorWidget _neuralNetWidget;
};
