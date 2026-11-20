#pragma once

#include "Definitions.h"

// What a workspace of this computer shows above its resources: its folder and, for the autosaves, the switch of the autosave
class _BrowserFolderBarWidget
{
public:
    static BrowserFolderBarWidget create();

    // The folder with its buttons, which takes the given width at most and returns the width it has taken
    float processFolder(float maxWidth);
    float getMinFolderWidth() const;

    void processAutosave();

private:
    _BrowserFolderBarWidget() = default;
};
