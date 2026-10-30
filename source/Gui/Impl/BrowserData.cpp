#include "BrowserData.h"

#include <algorithm>
#include <ranges>

#include <boost/algorithm/string/join.hpp>

#include <Base/Resources.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/TaskProcessor.h>

#include <Rendering/Interface/RenderingFacade.h>

#include "BrowserHelper.h"
#include "BrowserWindow.h"
#include "GenericMessageDialog.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"

BrowserData _BrowserData::create()
{
    return BrowserData(new _BrowserData());
}

_BrowserData::_BrowserData()
{
    _downloadCache = std::make_shared<_DownloadCache>();

    _reactionProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _emojiUserNameProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());

    auto numEmojis = 0;
    for (auto numEmojisPerBlock : NumEmojisPerBlock) {
        numEmojis += numEmojisPerBlock;
    }
    for (int i = 1; i <= numEmojis; ++i) {
        auto reactionName = "emoji" + std::to_string(i) + ".png";
        _emojis.emplace_back(_RenderingFacade::get()->loadTexture(Const::ImagesPath / std::filesystem::path(reactionName)));
    }

    for (NetworkResourceType resourceType = 0; resourceType < NetworkResourceType_Count; ++resourceType) {
        for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
            _workspaces.emplace(WorkspaceId{resourceType, workspaceType}, Workspace());
        }
    }
}

std::map<WorkspaceId, Workspace> const& _BrowserData::getWorkspaces() const
{
    return _workspaces;
}

Workspace const& _BrowserData::getWorkspace(WorkspaceId const& workspaceId) const
{
    return _workspaces.at(workspaceId);
}

Workspace const& _BrowserData::getCurrentWorkspace() const
{
    return _workspaces.at(_currentWorkspaceId);
}

WorkspaceId _BrowserData::getCurrentWorkspaceId() const
{
    return _currentWorkspaceId;
}

void _BrowserData::setCurrentWorkspaceId(WorkspaceId const& workspaceId)
{
    if (_currentWorkspaceId != workspaceId) {
        _currentWorkspaceId = workspaceId;
        _selectedTreeTO = nullptr;
    }
}

void _BrowserData::setRawTOs(WorkspaceId const& workspaceId, std::vector<NetworkResourceRawTO> const& rawTOs)
{
    _workspaces.at(workspaceId).rawTOs = rawTOs;
    createTreeTOs(workspaceId);
}

void _BrowserData::removeRawTOs(std::vector<NetworkResourceRawTO> const& rawTOs)
{
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        WorkspaceId workspaceId{_currentWorkspaceId.resourceType, workspaceType};
        auto& workspace = _workspaces.at(workspaceId);
        for (auto const& rawTO : rawTOs) {
            auto findResult = std::ranges::find_if(workspace.rawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
            if (findResult != workspace.rawTOs.end()) {
                workspace.rawTOs.erase(findResult);
            }
        }
        createTreeTOs(workspaceId);
    }
}

std::vector<NetworkResourceRawTO> _BrowserData::changeVisibility(std::vector<NetworkResourceRawTO> const& rawTOs)
{
    WorkspaceId publicWorkspaceId{_currentWorkspaceId.resourceType, WorkspaceType_Public};
    auto& publicRawTOs = _workspaces.at(publicWorkspaceId).rawTOs;

    std::vector<NetworkResourceRawTO> result;
    for (auto const& rawTO : rawTOs) {
        switch (rawTO->workspaceType) {
        case WorkspaceType_Private: {
            rawTO->workspaceType = WorkspaceType_Public;
            publicRawTOs.emplace_back(rawTO);
            result.emplace_back(rawTO);
        } break;
        case WorkspaceType_Public: {
            rawTO->workspaceType = WorkspaceType_Private;
            auto findResult = std::ranges::find_if(publicRawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
            if (findResult != publicRawTOs.end()) {
                publicRawTOs.erase(findResult);
            }
            result.emplace_back(rawTO);
        } break;
        default:
            break;
        }
    }
    if (!result.empty()) {
        createTreeTOs(publicWorkspaceId);
    }
    return result;
}

void _BrowserData::setSortSpecs(NetworkResourceType resourceType, std::vector<ImGuiTableColumnSortSpecs> const& sortSpecs)
{
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        WorkspaceId workspaceId{resourceType, workspaceType};
        _workspaces.at(workspaceId).sortSpecs = sortSpecs;
        createTreeTOs(workspaceId);
    }
}

void _BrowserData::setCollapsedFolderNames(WorkspaceId const& workspaceId, std::set<std::vector<std::string>> const& collapsedFolderNames)
{
    auto& workspace = _workspaces.at(workspaceId);
    workspace.collapsedFolderNames = collapsedFolderNames;
    createTreeTOs(workspaceId);

    if (workspaceId == _currentWorkspaceId) {
        deselectIfNotInTree(workspace);
    }
}

std::string const& _BrowserData::getFilter() const
{
    return _filter;
}

void _BrowserData::setFilter(std::string const& filter)
{
    _filter = filter;
    createTreeTOsForAllWorkspaces();
}

NetworkResourceTreeTO const& _BrowserData::getSelectedTreeTO() const
{
    return _selectedTreeTO;
}

void _BrowserData::setSelectedTreeTO(NetworkResourceTreeTO const& treeTO)
{
    _selectedTreeTO = treeTO;
}

bool _BrowserData::isScrollToSelection() const
{
    return _scrollToSelection;
}

void _BrowserData::setScrollToSelection(bool value)
{
    _scrollToSelection = value;
}

bool _BrowserData::isSelected(NetworkResourceRawTO const& rawTO) const
{
    return _selectedTreeTO != nullptr && _selectedTreeTO->isLeaf() && _selectedTreeTO->getLeaf().rawTO->id == rawTO->id;
}

bool _BrowserData::isSelected(NetworkResourceTreeTO const& treeTO) const
{
    if (_selectedTreeTO == nullptr || treeTO == nullptr) {
        return false;
    }
    if (treeTO->isLeaf()) {
        return isSelected(treeTO->getLeaf().rawTO);
    }
    return !_selectedTreeTO->isLeaf() && _selectedTreeTO->type == treeTO->type && _selectedTreeTO->folderNames == treeTO->folderNames;
}

bool _BrowserData::isOwner(NetworkResourceTreeTO const& treeTO) const
{
    if (treeTO == nullptr) {
        return false;
    }
    auto const& workspace = getCurrentWorkspace();

    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, workspace.rawTOs);
    auto userName = NetworkService::get().getLoggedInUserName().value_or("");
    return std::ranges::all_of(rawTOs, [&](NetworkResourceRawTO const& rawTO) { return rawTO->userName == userName; });
}

bool _BrowserData::isOwner(NetworkResourceRawTO const& rawTO) const
{
    return rawTO->userName == NetworkService::get().getLoggedInUserName().value_or("");
}

void _BrowserData::loadLastSessionData()
{
    _lastSessionData.load();
}

void _BrowserData::saveLastSessionData()
{
    _lastSessionData.save(getAllRawTOs());
}

void _BrowserData::registrateAsSeen(NetworkResourceRawTO const& rawTO)
{
    _lastSessionData.registrate(rawTO);
}

bool _BrowserData::isNew(NetworkResourceRawTO const& rawTO) const
{
    return !isOwner(rawTO) && _lastSessionData.isNew(rawTO);
}

std::vector<UserTO> const& _BrowserData::getUserTOs() const
{
    return _userTOs;
}

void _BrowserData::setUserTOs(std::vector<UserTO> const& userTOs)
{
    _userTOs = userTOs;
    std::sort(_userTOs.begin(), _userTOs.end(), [&](auto const& left, auto const& right) { return UserTO::compareOnlineAndTimestamp(left, right) > 0; });
}

std::optional<int> _BrowserData::getNumActiveSimulators() const
{
    return _numActiveSimulators;
}

void _BrowserData::setNumActiveSimulators(std::optional<int> const& numActiveSimulators)
{
    _numActiveSimulators = numActiveSimulators;
}

DownloadCache const& _BrowserData::getDownloadCache() const
{
    return _downloadCache;
}

void _BrowserData::onDownloadResource(BrowserLeaf const& leaf)
{
    ++leaf.rawTO->numDownloads;

    NetworkTransferController::get().onDownload(DownloadNetworkResourceRequestData{
        .resourceId = leaf.rawTO->id,
        .resourceName = leaf.rawTO->resourceName,
        .resourceVersion = leaf.rawTO->version,
        .resourceType = _currentWorkspaceId.resourceType,
        .downloadCache = _downloadCache});
}

std::vector<TextureData> const& _BrowserData::getEmojis() const
{
    return _emojis;
}

void _BrowserData::setOwnEmojiTypeBySimId(std::unordered_map<std::string, int> const& ownEmojiTypeBySimId)
{
    _ownEmojiTypeBySimId = ownEmojiTypeBySimId;
}

bool _BrowserData::isOwnReaction(std::string const& resourceId, int emojiType) const
{
    auto findResult = _ownEmojiTypeBySimId.find(resourceId);
    return findResult != _ownEmojiTypeBySimId.end() && findResult->second == emojiType;
}

std::string _BrowserData::getUserNamesToEmojiType(std::string const& resourceId, int emojiType)
{
    std::set<std::string> userNames;

    auto findResult = _userNamesByEmojiTypeBySimIdCache.find(std::make_pair(resourceId, emojiType));
    if (findResult != _userNamesByEmojiTypeBySimIdCache.end()) {
        userNames = findResult->second;
    } else {
        if (!_emojiUserNameProcessor->pendingTasks()) {
            _emojiUserNameProcessor->executeTask(
                [&](auto const& senderId) {
                    return _PersisterFacade::get()->scheduleGetUserNamesForReaction(
                        SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = false},
                        GetUserNamesForReactionRequestData{.resourceId = resourceId, .emojiType = emojiType});
                },
                [&](auto const& requestId) {
                    auto data = _PersisterFacade::get()->fetchGetUserNamesForReactionData(requestId);
                    _userNamesByEmojiTypeBySimIdCache.emplace(std::make_pair(data.resourceId, data.emojiType), data.userNames);
                },
                [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
        }
        return "Loading...";
    }

    return boost::algorithm::join(userNames, ", ");
}

void _BrowserData::onToggleLike(NetworkResourceTreeTO const& treeTO, int emojiType)
{
    CHECK(treeTO->isLeaf());
    auto& leaf = treeTO->getLeaf();
    if (NetworkService::get().getLoggedInUserName()) {

        // Remove existing like
        auto findResult = _ownEmojiTypeBySimId.find(leaf.rawTO->id);
        auto onlyRemoveLike = false;
        if (findResult != _ownEmojiTypeBySimId.end()) {
            auto origEmojiType = findResult->second;
            if (--leaf.rawTO->numLikesByEmojiType[origEmojiType] == 0) {
                leaf.rawTO->numLikesByEmojiType.erase(origEmojiType);
            }
            _ownEmojiTypeBySimId.erase(findResult);
            _userNamesByEmojiTypeBySimIdCache.erase(std::make_pair(leaf.rawTO->id, origEmojiType));  // Invalidate cache entry
            onlyRemoveLike = origEmojiType == emojiType;                                             // Remove like if same like icon has been clicked
        }

        // Create new like
        if (!onlyRemoveLike) {
            _ownEmojiTypeBySimId[leaf.rawTO->id] = emojiType;
            if (leaf.rawTO->numLikesByEmojiType.contains(emojiType)) {
                ++leaf.rawTO->numLikesByEmojiType[emojiType];
            } else {
                leaf.rawTO->numLikesByEmojiType[emojiType] = 1;
            }
        }

        _userNamesByEmojiTypeBySimIdCache.erase(std::make_pair(leaf.rawTO->id, emojiType));  // Invalidate cache entry

        _reactionProcessor->executeTask(
            [&](auto const& senderId) {
                return _PersisterFacade::get()->scheduleToggleReactionNetworkResource(
                    SenderInfo{.senderId = senderId, .wishResultData = false, .wishErrorInfo = false},
                    ToggleReactionNetworkResourceRequestData{.resourceId = leaf.rawTO->id, .emojiType = emojiType});
            },
            [](auto const&) {},
            [](auto const& errors) {
                BrowserWindow::get().onRefresh();
                GenericMessageDialog::get().information("Error", errors);
            });
    } else {
        LoginDialog::get().open();
    }
}

bool _BrowserData::isActivateEmojiPopup() const
{
    return _activateEmojiPopup;
}

void _BrowserData::setActivateEmojiPopup(bool value)
{
    _activateEmojiPopup = value;
}

NetworkResourceTreeTO const& _BrowserData::getEmojiPopupTO() const
{
    return _emojiPopupTO;
}

void _BrowserData::setEmojiPopupTO(NetworkResourceTreeTO const& treeTO)
{
    _emojiPopupTO = treeTO;
}

void _BrowserData::processPendingRequests()
{
    _reactionProcessor->process();
    _emojiUserNameProcessor->process();
}

void _BrowserData::createTreeTOs(WorkspaceId const& workspaceId)
{
    auto& workspace = _workspaces.at(workspaceId);

    // Sorting
    if (workspace.rawTOs.size() > 1) {
        std::sort(workspace.rawTOs.begin(), workspace.rawTOs.end(), [&](auto const& left, auto const& right) {
            return _NetworkResourceRawTO::compare(left, right, workspace.sortSpecs) < 0;
        });
    }

    // Filtering
    std::vector<NetworkResourceRawTO> filteredRawTOs;
    for (auto const& rawTO : workspace.rawTOs) {
        if (rawTO->matchWithFilter(_filter)) {
            filteredRawTOs.emplace_back(rawTO);
        }
    }

    // Create treeTOs
    workspace.treeTOs = NetworkResourceService::get().createTreeTOs(filteredRawTOs, workspace.collapsedFolderNames);

    if (workspaceId == _currentWorkspaceId) {
        updateSelection(workspace, filteredRawTOs);
    }
}

void _BrowserData::createTreeTOsForAllWorkspaces()
{
    for (auto const& workspaceId : _workspaces | std::views::keys) {
        createTreeTOs(workspaceId);
    }
}

void _BrowserData::updateSelection(Workspace const& workspace, std::vector<NetworkResourceRawTO> const& filteredRawTOs)
{
    if (_selectedTreeTO == nullptr) {
        return;
    }

    // The tree objects are rebuilt and a refresh also replaces the raw objects, so the selection is pointed to the new ones
    if (_selectedTreeTO->isLeaf()) {

        // Searched in the raw objects since the tree omits the resources of collapsed folders, which the gallery still shows
        auto findResult = std::ranges::find_if(filteredRawTOs, [this](NetworkResourceRawTO const& rawTO) { return isSelected(rawTO); });
        _selectedTreeTO = findResult != filteredRawTOs.end() ? BrowserHelper::createLeafTreeTO(*findResult) : nullptr;
    } else {
        auto findResult = std::ranges::find_if(workspace.treeTOs, [this](NetworkResourceTreeTO const& treeTO) { return isSelected(treeTO); });
        _selectedTreeTO = findResult != workspace.treeTOs.end() ? *findResult : nullptr;
    }
}

void _BrowserData::deselectIfNotInTree(Workspace const& workspace)
{
    if (std::ranges::none_of(workspace.treeTOs, [this](NetworkResourceTreeTO const& treeTO) { return isSelected(treeTO); })) {
        _selectedTreeTO = nullptr;
    }
}

std::unordered_set<NetworkResourceRawTO> _BrowserData::getAllRawTOs() const
{
    std::unordered_set<NetworkResourceRawTO> result;
    for (auto const& workspace : _workspaces | std::views::values) {
        result.insert(workspace.rawTOs.begin(), workspace.rawTOs.end());
    }
    return result;
}
