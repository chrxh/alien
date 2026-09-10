#pragma once

#include <PersisterInterface/SavepointTable.h>

#include "AlienWindow.h"
#include "AutosaveController.h"
#include "Definitions.h"

class AutosaveWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(AutosaveWindow);

private:
    AutosaveWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;

    void processToolbar();
    void processHeader();
    void processTable();
    void processSettings();
    void processStatusBar();

    void onLoadSavepoint(SavepointEntry const& entry);

    bool _settingsOpen = false;
    float _settingsHeight = 0;

    std::string _origDirectory;
    int _origAutosaveInterval = 40;
    AutosaveController::SaveMode _origSaveMode = AutosaveController::SaveMode_Circular;
    int _origNumberOfFiles = 20;

    SavepointEntry _selectedEntry;
};
