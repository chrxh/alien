#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include <Network/NetworkResourceRawTO.h>
#include <Network/NetworkResourceTreeTO.h>

#include <Persister/Interface/SerializerService.h>

#include "Definitions.h"

// Shows the resource that is selected in the browser with its picture, its description, its facts and everything that can be done with it.
// Without a selection the current workspace is described.
class _BrowserDetailsWidget
{
public:
    struct Actions
    {
        std::function<void(NetworkResourceTreeTO const&)> edit;
        std::function<void(NetworkResourceTreeTO const&)> changeVisibility;
        std::function<void(NetworkResourceTreeTO const&)> remove;
        std::function<void(NetworkResourceRawTO const&)> share;    // Uploads a simulation or a genome of this computer
        std::function<void(NetworkResourceRawTO const&)> replace;  // Replaces the content of an own upload
    };
    static BrowserDetailsWidget create(BrowserData const& data, BrowserPictureCache const& pictureCache, Actions const& actions);

    void process(float width, float height);

private:
    _BrowserDetailsWidget(BrowserData const& data, BrowserPictureCache const& pictureCache, Actions const& actions);

    void processResource(NetworkResourceTreeTO const& treeTO);
    void processFacts(NetworkResourceRawTO const& rawTO);
    void processReactions(NetworkResourceTreeTO const& treeTO);
    void processVisibility(NetworkResourceTreeTO const& treeTO);
    void processResourceButtons(NetworkResourceTreeTO const& treeTO);

    void processFolder(NetworkResourceTreeTO const& treeTO);
    void processWorkspace();
    void processFolders();
    void processAutosaveFolder();
    void processAutosaveSettings();

    std::optional<SimulationFileInfo> const& getFileInfo(NetworkResourceRawTO const& rawTO);

    BrowserData _data;
    BrowserPictureCache _pictureCache;
    Actions _actions;

    std::unordered_map<std::string, std::optional<SimulationFileInfo>> _fileInfoByKey;
    std::optional<int> _libraryRevision;

    std::string _autosaveFolderName;
    int _lastAutosaveFolderFrame = -1;
};
