#pragma once

#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _EditSimulationDialog : public AlienDialog
{
public:
    _EditSimulationDialog();
    virtual ~_EditSimulationDialog() override = default;

    void openForLeaf(NetworkResourceTreeTO const& treeTO);
    void openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);

private:
    void processIntern();

    void processForLeaf();
    void processForFolder();

    NetworkResourceTreeTO _treeTO;
    std::vector<NetworkResourceRawTO> _rawTOs;
    std::string _origFolderName;
    std::string _newName;
    std::string _newDescription;
};