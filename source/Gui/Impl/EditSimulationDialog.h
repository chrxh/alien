#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <Network/Definitions.h>

#include "AlienDialog.h"
#include "Definitions.h"

// Changes the name and the description of a resource of the browser or the name of one of its folders, on the server and on this computer
class EditSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EditSimulationDialog);

public:
    void openForLeaf(NetworkResourceTreeTO const& treeTO);
    void openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);

private:
    EditSimulationDialog();

    void processIntern() override;

    void processForLeaf();
    void processForFolder();
    void processErrorMessage();  // Below the name it refers to
    void processButtons(bool valid);

    bool isLocal() const;
    void onApplyToLeaf();
    void onApplyToFolder();

    NetworkResourceTreeTO _treeTO;
    std::vector<NetworkResourceRawTO> _rawTOs;
    std::string _origFolderName;
    std::string _newName;
    std::string _newDescription;
    bool _descriptionEditable = true;
    std::optional<std::string> _errorMessage;
};
