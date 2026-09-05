#include "BrowserData.h"

#include <algorithm>
#include <ranges>

#include <boost/algorithm/string/join.hpp>

#include <Base/Resources.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/TaskProcessor.h>

#include "BrowserWindow.h"
#include "GenericMessageDialog.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "OpenGLHelper.h"

BrowserData _BrowserData::create()
{
    return BrowserData(new _BrowserData());
}

_BrowserData::_BrowserData()
{
    downloadCache = std::make_shared<_DownloadCache>();

    _reactionProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _emojiUserNameProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());

    auto numEmojis = 0;
    for (auto numEmojisPerBlock : NumEmojisPerBlock) {
        numEmojis += numEmojisPerBlock;
    }
    for (int i = 1; i <= numEmojis; ++i) {
        auto reactionName = "emoji" + std::to_string(i) + ".png";
        emojis.emplace_back(OpenGLHelper::loadTexture(Const::ImagesPath / std::filesystem::path(reactionName)));
    }

    for (NetworkResourceType resourceType = 0; resourceType < NetworkResourceType_Count; ++resourceType) {
        for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
            workspaces.emplace(WorkspaceId{resourceType, workspaceType}, Workspace());
        }
    }
}

Workspace& _BrowserData::getCurrentWorkspace()
{
    return workspaces.at(currentWorkspace);
}

void _BrowserData::createTreeTOs(Workspace& workspace)
{
    // Sorting
    if (workspace.rawTOs.size() > 1) {
        std::sort(workspace.rawTOs.begin(), workspace.rawTOs.end(), [&](auto const& left, auto const& right) {
            return _NetworkResourceRawTO::compare(left, right, workspace.sortSpecs) < 0;
        });
    }

    // Filtering
    std::vector<NetworkResourceRawTO> filteredRawTOs;
    for (auto const& rawTO : workspace.rawTOs) {
        if (rawTO->matchWithFilter(filter)) {
            filteredRawTOs.emplace_back(rawTO);
        }
    }

    // Create treeTOs
    workspace.treeTOs = NetworkResourceService::get().createTreeTOs(filteredRawTOs, workspace.collapsedFolderNames);
    selectedTreeTO = nullptr;
}

void _BrowserData::createTreeTOsForAllWorkspaces()
{
    for (auto& workspace : workspaces | std::views::values) {
        createTreeTOs(workspace);
    }
}

void _BrowserData::createTreeTOsForCurrentResourceType()
{
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        createTreeTOs(workspaces.at(WorkspaceId{currentWorkspace.resourceType, workspaceType}));
    }
}

std::unordered_set<NetworkResourceRawTO> _BrowserData::getAllRawTOs() const
{
    std::unordered_set<NetworkResourceRawTO> result;
    for (auto const& workspace : workspaces | std::views::values) {
        result.insert(workspace.rawTOs.begin(), workspace.rawTOs.end());
    }
    return result;
}

bool _BrowserData::isOwner(NetworkResourceTreeTO const& treeTO) const
{
    if (treeTO == nullptr) {
        return false;
    }
    auto const& workspace = workspaces.at(currentWorkspace);

    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, workspace.rawTOs);
    auto userName = NetworkService::get().getLoggedInUserName().value_or("");
    return std::ranges::all_of(rawTOs, [&](NetworkResourceRawTO const& rawTO) { return rawTO->userName == userName; });
}

bool _BrowserData::isSelected(NetworkResourceRawTO const& rawTO) const
{
    return selectedTreeTO != nullptr && selectedTreeTO->isLeaf() && selectedTreeTO->getLeaf().rawTO->id == rawTO->id;
}

bool _BrowserData::isSelected(NetworkResourceTreeTO const& treeTO) const
{
    if (selectedTreeTO == nullptr || treeTO == nullptr) {
        return false;
    }
    if (treeTO->isLeaf()) {
        return isSelected(treeTO->getLeaf().rawTO);
    }
    return !selectedTreeTO->isLeaf() && selectedTreeTO->type == treeTO->type && selectedTreeTO->folderNames == treeTO->folderNames;
}

void _BrowserData::sortUserList()
{
    std::sort(userTOs.begin(), userTOs.end(), [&](auto const& left, auto const& right) { return UserTO::compareOnlineAndTimestamp(left, right) > 0; });
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

void _BrowserData::onDownloadResource(BrowserLeaf const& leaf)
{
    ++leaf.rawTO->numDownloads;

    NetworkTransferController::get().onDownload(DownloadNetworkResourceRequestData{
        .resourceId = leaf.rawTO->id,
        .resourceName = leaf.rawTO->resourceName,
        .resourceVersion = leaf.rawTO->version,
        .resourceType = currentWorkspace.resourceType,
        .downloadCache = downloadCache});
}

void _BrowserData::onToggleLike(NetworkResourceTreeTO const& treeTO, int emojiType)
{
    CHECK(treeTO->isLeaf());
    auto& leaf = treeTO->getLeaf();
    if (NetworkService::get().getLoggedInUserName()) {

        // Remove existing like
        auto findResult = ownEmojiTypeBySimId.find(leaf.rawTO->id);
        auto onlyRemoveLike = false;
        if (findResult != ownEmojiTypeBySimId.end()) {
            auto origEmojiType = findResult->second;
            if (--leaf.rawTO->numLikesByEmojiType[origEmojiType] == 0) {
                leaf.rawTO->numLikesByEmojiType.erase(origEmojiType);
            }
            ownEmojiTypeBySimId.erase(findResult);
            _userNamesByEmojiTypeBySimIdCache.erase(std::make_pair(leaf.rawTO->id, origEmojiType));  // Invalidate cache entry
            onlyRemoveLike = origEmojiType == emojiType;                                             // Remove like if same like icon has been clicked
        }

        // Create new like
        if (!onlyRemoveLike) {
            ownEmojiTypeBySimId[leaf.rawTO->id] = emojiType;
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

void _BrowserData::processPendingRequests()
{
    _reactionProcessor->process();
    _emojiUserNameProcessor->process();
}
