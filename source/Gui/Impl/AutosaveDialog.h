#pragma once

#include <filesystem>
#include <string>

#include <Base/Singleton.h>

#include "AlienDialog.h"
#include "Definitions.h"

// Asks how the simulation is saved automatically and switches the autosave on
class AutosaveDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(AutosaveDialog);

private:
    AutosaveDialog();

    void openIntern() override;
    void processIntern() override;

    void processSettings(bool subfolderValid);
    void processSummary(std::filesystem::path const& folder);

    void onActivate();

    int _interval = 1;
    std::string _folderName;
    int _saveMode = 0;
    int _numberOfFiles = 1;
};
