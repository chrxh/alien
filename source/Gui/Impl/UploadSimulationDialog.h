#pragma once

#include <optional>
#include <string>

#include <Base/Singleton.h>

#include <Network/Definitions.h>

#include <Engine/Interface/Definitions.h>

#include "AlienDialog.h"
#include "Definitions.h"
#include "ResourcePreviewWidget.h"

class UploadSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(UploadSimulationDialog);

public:
    void open(NetworkResourceType resourceType, std::string const& folder = "", std::string const& resourceName = "");

private:
    UploadSimulationDialog();

    void processIntern() override;

    void onUpload(WorkspaceType workspaceType);

    std::string _folder;
    std::unordered_map<std::string, std::string> _resourceDescriptionByFolder;

    std::string _resourceName;
    std::string _resourceDescription;

    NetworkResourceType _resourceType = NetworkResourceType_Simulation;

    ResourcePreviewWidget _preview;
};
