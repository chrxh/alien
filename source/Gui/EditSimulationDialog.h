#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _EditSimulationDialog : public _AlienDialog
{
public:
    _EditSimulationDialog(BrowserWindow const& browserWindow);
    virtual ~_EditSimulationDialog() override = default;

    void openForLeaf(NetworkResourceTreeTO const& treeTO);
    void openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);

private:
    void processIntern();

    void processLeaf();

    NetworkResourceTreeTO _treeTO;
    std::vector<NetworkResourceRawTO> _rawTOs;
    std::string _newName;
    std::string _newDescription;

    BrowserWindow _browserWindow;
};