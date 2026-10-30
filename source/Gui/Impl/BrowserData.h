#pragma once

#include <map>
#include <optional>
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

#include <Persister/Interface/Definitions.h>
#include <Persister/Interface/DownloadCache.h>

#include <Rendering/Interface/TextureData.h>

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

    std::map<WorkspaceId, Workspace> const& getWorkspaces() const;
    Workspace const& getWorkspace(WorkspaceId const& workspaceId) const;
    Workspace const& getCurrentWorkspace() const;
    WorkspaceId getCurrentWorkspaceId() const;
    void setCurrentWorkspaceId(WorkspaceId const& workspaceId);
    void setRawTOs(WorkspaceId const& workspaceId, std::vector<NetworkResourceRawTO> const& rawTOs);
    void removeRawTOs(std::vector<NetworkResourceRawTO> const& rawTOs);
    std::vector<NetworkResourceRawTO> changeVisibility(std::vector<NetworkResourceRawTO> const& rawTOs);
    void setSortSpecs(NetworkResourceType resourceType, std::vector<ImGuiTableColumnSortSpecs> const& sortSpecs);
    void setCollapsedFolderNames(WorkspaceId const& workspaceId, std::set<std::vector<std::string>> const& collapsedFolderNames);

    std::string const& getFilter() const;
    void setFilter(std::string const& filter);

    NetworkResourceTreeTO const& getSelectedTreeTO() const;
    void setSelectedTreeTO(NetworkResourceTreeTO const& treeTO);
    bool isScrollToSelection() const;
    void setScrollToSelection(bool value);

    // The gallery and the table build their own tree objects, so the selection is compared by resource and not by object identity
    bool isSelected(NetworkResourceRawTO const& rawTO) const;
    bool isSelected(NetworkResourceTreeTO const& treeTO) const;

    bool isOwner(NetworkResourceTreeTO const& treeTO) const;
    bool isOwner(NetworkResourceRawTO const& rawTO) const;

    void loadLastSessionData();
    void saveLastSessionData();
    void registrateAsSeen(NetworkResourceRawTO const& rawTO);
    bool isNew(NetworkResourceRawTO const& rawTO) const;

    std::vector<UserTO> const& getUserTOs() const;
    void setUserTOs(std::vector<UserTO> const& userTOs);
    std::optional<int> getNumActiveSimulators() const;
    void setNumActiveSimulators(std::optional<int> const& numActiveSimulators);

    DownloadCache const& getDownloadCache() const;
    void onDownloadResource(BrowserLeaf const& leaf);

    std::vector<TextureData> const& getEmojis() const;
    void setOwnEmojiTypeBySimId(std::unordered_map<std::string, int> const& ownEmojiTypeBySimId);
    bool isOwnReaction(std::string const& resourceId, int emojiType) const;
    std::string getUserNamesToEmojiType(std::string const& resourceId, int emojiType);
    void onToggleLike(NetworkResourceTreeTO const& treeTO, int emojiType);

    bool isActivateEmojiPopup() const;
    void setActivateEmojiPopup(bool value);
    NetworkResourceTreeTO const& getEmojiPopupTO() const;
    void setEmojiPopupTO(NetworkResourceTreeTO const& treeTO);

    void processPendingRequests();

private:
    _BrowserData();

    void createTreeTOs(WorkspaceId const& workspaceId);
    void createTreeTOsForAllWorkspaces();
    void updateSelection(Workspace const& workspace, std::vector<NetworkResourceRawTO> const& filteredRawTOs);
    void deselectIfNotInTree(Workspace const& workspace);
    std::unordered_set<NetworkResourceRawTO> getAllRawTOs() const;

    std::map<WorkspaceId, Workspace> _workspaces;
    WorkspaceId _currentWorkspaceId = {NetworkResourceType_Simulation, WorkspaceType_AlienProject};
    std::string _filter;
    NetworkResourceTreeTO _selectedTreeTO;
    bool _scrollToSelection = false;
    LastSessionBrowserData _lastSessionData;
    std::vector<UserTO> _userTOs;
    std::optional<int> _numActiveSimulators;
    DownloadCache _downloadCache;

    std::vector<TextureData> _emojis;
    std::unordered_map<std::string, int> _ownEmojiTypeBySimId;
    bool _activateEmojiPopup = false;
    NetworkResourceTreeTO _emojiPopupTO;

    TaskProcessor _reactionProcessor;
    TaskProcessor _emojiUserNameProcessor;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;
};
