#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <imgui.h>

#include <Base/Hashes.h>

#include <Network/NetworkResourceRawTO.h>
#include <Network/NetworkResourceTreeTO.h>
#include <Network/UserTO.h>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/DownloadCache.h>

#include "Definitions.h"
#include "LastSessionBrowserData.h"

struct WorkspaceId
{
    NetworkResourceType resourceType;
    WorkspaceType workspaceType;
    auto operator<=>(WorkspaceId const&) const = default;
};

struct Workspace
{
    std::vector<ImGuiTableColumnSortSpecs> sortSpecs;
    std::vector<NetworkResourceRawTO> rawTOs;    // Unfiltered, sorted
    std::vector<NetworkResourceTreeTO> treeTOs;  // Filtered, sorted
    std::set<std::vector<std::string>> collapsedFolderNames;
};

// Browser state shared between the window and its widgets
class _BrowserData
{
public:
    static auto constexpr NumEmojiBlocks = 4;
    static constexpr int NumEmojisPerBlock[NumEmojiBlocks] = {19, 14, 10, 6};
    static auto constexpr NumEmojisPerRow = 5;

    static BrowserData create();

    std::map<WorkspaceId, Workspace> workspaces;
    WorkspaceId currentWorkspace = {NetworkResourceType_Simulation, WorkspaceType_AlienProject};
    NetworkResourceTreeTO selectedTreeTO;
    std::string filter;
    std::vector<UserTO> userTOs;
    LastSessionBrowserData lastSessionData;
    DownloadCache downloadCache;

    std::vector<TextureData> emojis;
    std::unordered_map<std::string, int> ownEmojiTypeBySimId;
    bool activateEmojiPopup = false;
    NetworkResourceTreeTO emojiPopupTO;

    Workspace& getCurrentWorkspace();
    void createTreeTOs(Workspace& workspace);
    void createTreeTOsForAllWorkspaces();
    void createTreeTOsForCurrentResourceType();
    std::unordered_set<NetworkResourceRawTO> getAllRawTOs() const;
    bool isOwner(NetworkResourceTreeTO const& treeTO) const;

    // The gallery and the table build their own tree objects, so the selection is compared by resource and not by object identity
    bool isSelected(NetworkResourceRawTO const& rawTO) const;
    bool isSelected(NetworkResourceTreeTO const& treeTO) const;

    void sortUserList();

    std::string getUserNamesToEmojiType(std::string const& resourceId, int emojiType);

    void onDownloadResource(BrowserLeaf const& leaf);
    void onToggleLike(NetworkResourceTreeTO const& treeTO, int emojiType);

    void processPendingRequests();

private:
    _BrowserData();

    TaskProcessor _reactionProcessor;
    TaskProcessor _emojiUserNameProcessor;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;
};
