#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _EditSimulationDialog : public _AlienDialog
{
public:
    _EditSimulationDialog(BrowserWindow const& browserWindow);
    ~_EditSimulationDialog();

    void open(NetworkResourceTreeTO const& treeTO);

private:
    void processIntern();

    NetworkResourceTreeTO _treeTO;
    BrowserWindow _browserWindow;
};