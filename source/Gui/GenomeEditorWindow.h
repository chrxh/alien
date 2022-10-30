#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "AlienWindow.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow();
    ~_GenomeEditorWindow() override;

private:
    void processIntern() override;

    struct TabData
    {
        GenomeDescription genome;
    };
    std::vector<TabData> _tabDatas;
};
