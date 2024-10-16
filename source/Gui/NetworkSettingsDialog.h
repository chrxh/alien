#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _NetworkSettingsDialog : public AlienDialog
{
public:
    _NetworkSettingsDialog();

private:
    void processIntern();
    void openIntern();

    void onChangeSettings();

    std::string _serverAddress;
    std::string _origServerAddress;
};