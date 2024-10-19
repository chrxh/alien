#pragma once

#include "Base/Singleton.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class EditSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EditSimulationDialog);

public:
    void openForLeaf(NetworkResourceTreeTO const& treeTO);
    void openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);

private:
    EditSimulationDialog();

    void processIntern();

    void processForLeaf();
    void processForFolder();

    NetworkResourceTreeTO _treeTO;
    std::vector<NetworkResourceRawTO> _rawTOs;
    std::string _origFolderName;
    std::string _newName;
    std::string _newDescription;
};