#pragma once

#include "Definitions.h"

// The row above the resources of a workspace of this computer: its folder and, for the autosaves, the switch of the autosave
class _BrowserFolderBarWidget
{
public:
    static BrowserFolderBarWidget create(BrowserData const& data);

    void process();

private:
    _BrowserFolderBarWidget(BrowserData const& data);

    void processFolder(float fieldWidth);
    void processAutosave(bool showInfo, bool showInterval);

    BrowserData _data;
};
