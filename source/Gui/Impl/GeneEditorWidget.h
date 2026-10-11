#pragma once

#include <Engine/Interface/Definitions.h>

#include "Definitions.h"

class _GeneEditorWidget
{
public:
    static GeneEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _GeneEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void processNoSelection();
    void processHeaderData();

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;
};
