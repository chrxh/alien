#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <Data/GenomeDesc.h>

#include <Network/Definitions.h>

#include <Engine/Interface/Definitions.h>

#include <Persister/Interface/SimulationLibrary.h>

#include "AlienDialog.h"
#include "Definitions.h"
#include "ResourcePreviewWidget.h"

enum class SaveTarget
{
    ThisComputer,
    Community,
    Private,
    OtherLocation  // A file in any folder, which a file dialog asks for
};

// Stores a simulation or a genome under a name with a description and a picture: on this computer, or on the server as an upload. A name that is
// already in use at the target replaces what is stored there. The dialog also leads to the file dialog that saves to any folder.
class SaveSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SaveSimulationDialog);

public:
    // Saves the running simulation or the genome of the genome editor. A suggested name that ends with a slash names only the folder.
    void open(NetworkResourceType resourceType, SaveTarget target, std::string const& suggestedName = "");

    // Uploads a simulation or a genome of this computer
    void openForSharing(NetworkResourceType resourceType, std::filesystem::path const& file, std::string const& name, std::string const& description);

    // Replaces the content and the picture of an own upload by the running simulation or the genome of the genome editor
    void openForReplacing(NetworkResourceRawTO const& upload);

private:
    SaveSimulationDialog();

    enum class Mode
    {
        RunningResource,
        Sharing,
        Replacing
    };

    // What the dialog does with the current input
    struct Outcome
    {
        bool possible = false;
        bool needsLogin = false;
        std::optional<std::string> replacedResourceId;  // An own upload with the same name at the target
        bool replacesFile = false;
        std::optional<std::string> message;
    };

    // What the description field has been filled for
    struct DescriptionSource
    {
        SaveTarget target = SaveTarget::ThisComputer;
        std::string name;
        bool replacesFile = false;
        std::optional<std::string> replacedResourceId;
        bool operator==(DescriptionSource const&) const = default;
    };

    void processIntern() override;
    void processForm();
    void processFooter(Outcome const& outcome);

    Outcome evaluate() const;
    void resetDescription(std::string const& description);
    void updateDescription(Outcome const& outcome);
    std::optional<SimulationDescription> findReplacedDescription(Outcome const& outcome) const;
    void onConfirm(Outcome const& outcome);
    void onUpload();
    void onReplaceUpload(std::string const& resourceId, WorkspaceType workspaceType);

    Mode _mode = Mode::RunningResource;
    NetworkResourceType _resourceType = NetworkResourceType_Simulation;
    std::vector<SaveTarget> _targets;
    SaveTarget _target = SaveTarget::ThisComputer;

    std::string _name;
    std::string _description;
    std::string _ownDescription;  // Shown for a name that is not in use: what the dialog started with, or what has been typed
    bool _descriptionEdited = false;
    bool _descriptionEditable = true;
    std::optional<DescriptionSource> _descriptionSource;
    std::map<std::pair<NetworkResourceType, std::string>, std::string> _descriptionByTypeAndFolder;
    std::optional<std::string> _errorMessage;

    std::optional<std::filesystem::path> _simulationFile;  // A saved simulation instead of the running one
    std::optional<GenomeDesc> _savedGenome;                // A saved genome instead of the one of the genome editor
    NetworkResourceRawTO _replacedUpload;
    ResourcePreviewWidget _preview;
};
