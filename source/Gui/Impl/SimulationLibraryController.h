#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <Data/GenomeDesc.h>

#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/SimulationLibrary.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

// Keeps the simulations and genomes of this computer: saves the running simulation by hand and automatically, loads simulations, saves
// genomes and manages the saved files
class SimulationLibraryController : public MainLoopEntity
{
    MAKE_SINGLETON(SimulationLibraryController);

public:
    using SaveMode = int;
    enum SaveMode_
    {
        SaveMode_Limited,
        SaveMode_Unlimited
    };

    struct Renaming
    {
        std::filesystem::path simulationPath;
        std::string newName;
    };

    void process() override;

    std::filesystem::path const& getLibraryFolder() const;
    void setLibraryFolder(std::filesystem::path const& value);

    // The autosaves lie in the library folder until they get a folder of their own
    std::filesystem::path const& getAutosaveMainFolder() const;
    void setAutosaveMainFolder(std::filesystem::path const& value);

    // The library folder or the folder of the autosaves, which the name of the simulation is relative to
    std::filesystem::path const& getMainFolder(std::filesystem::path const& simulationPath) const;

    SimulationLibraryContent const& getContent() const;
    int getRevision() const;  // Changes with every change of the content
    void refresh();           // Reads the content again, for instance after other programs may have changed it

    bool isAutosaveEnabled() const;
    void setAutosaveEnabled(bool value);
    int getAutosaveInterval() const;
    void setAutosaveInterval(int value);
    SaveMode getSaveMode() const;
    void setSaveMode(SaveMode value);
    int getNumberOfFiles() const;
    void setNumberOfFiles(int value);
    std::string const& getAutosaveFolderName() const;                  // Folder inside the library, empty for the library folder itself
    bool setAutosaveFolderName(std::string const& value);              // Returns false for a name that is not valid
    bool isAutosaveFolder(std::filesystem::path const& folder) const;  // The folder in which autosave stores its files and limits their number
    std::chrono::seconds getDurationUntilNextAutosave() const;

    bool isSaving() const;

    // Without a description, the description of a replaced simulation is kept
    void onSave(std::string const& name, std::optional<std::string> const& description, std::string const& jpg);
    void onSaveToFile(std::filesystem::path const& filename);  // A file at any place
    void onAutosave();
    void onLoad(std::filesystem::path const& simulationPath);

    // These functions return an error message if they fail. Editing, renaming and deleting take simulations and genomes.
    // Without a description, the description of a replaced genome is kept.
    std::optional<std::string>
    onSaveGenome(std::string const& name, GenomeDesc const& genome, std::optional<std::string> const& description, std::string const& jpg);
    std::optional<std::string>
    onEdit(std::filesystem::path const& simulationPath, std::string const& newName, std::optional<std::string> const& newDescription);
    std::optional<std::string> onRename(std::vector<Renaming> const& renamings);
    std::optional<std::string> onDelete(std::vector<std::filesystem::path> const& simulationPaths);

    std::optional<std::string> fetchSavedFilename();

private:
    void init() override;
    void shutdown() override;

    struct PendingSave
    {
        PersisterRequestId requestId;
        std::filesystem::path target;     // The file, or the folder of an autosave
        std::optional<std::string> name;  // Name inside the library of a simulation saved by hand
        std::optional<std::string> description;
        std::optional<std::string> jpg;
        bool automatic = false;
        int sessionId = 0;  // Tells whether the saved simulation is still the running one when it has been saved
    };
    struct PendingLoad
    {
        PersisterRequestId requestId;
        std::filesystem::path simulationPath;
    };
    struct PendingPicture
    {
        std::filesystem::path simulationPath;
        int sessionId = 0;
        int numFramesToWait = 0;
    };

    void processPendingSaves();
    void processPendingLoads();
    void processPendingPicture();
    void processAutosave();

    void scheduleSave(PendingSave pendingSave);
    void onSaveFinished(PendingSave const& pendingSave, SaveSimulationResultData const& data);
    std::filesystem::path getAutosaveFolder() const;
    std::optional<std::string> rename(std::filesystem::path const& simulationPath, std::string const& newName, std::filesystem::path& newSimulationPath);
    bool isTransferring(std::filesystem::path const& simulationPath) const;

    std::filesystem::path _libraryFolder;
    std::optional<std::filesystem::path> _autosaveMainFolder;
    SimulationLibraryContent _content;
    bool _contentOutdated = true;
    int _revision = 0;

    bool _autosaveEnabled = false;
    int _autosaveInterval = 40;
    SaveMode _saveMode = SaveMode_Limited;
    int _numberOfFiles = 20;
    std::string _autosaveFolderName;

    std::vector<PendingSave> _pendingSaves;
    std::vector<PendingLoad> _pendingLoads;
    std::optional<PendingPicture> _pendingPicture;
    std::optional<std::string> _savedFilename;

    std::chrono::steady_clock::time_point _lastAutosaveTimepoint;
    std::optional<int> _lastSessionId;
};
