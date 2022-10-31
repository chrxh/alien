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

    void showToolbar();

    struct TabData
    {
        GenomeDescription genome;
        std::optional<int> selected;
    };
    void showGenomeContent(TabData& tabData);

    std::vector<TabData> _tabDatas;
    int _currentTabIndex = 0;
    float _previewHeight = 200.0f;
};
