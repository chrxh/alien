#include "BrowserController.h"

#include <optional>

#include <Network/NetworkService.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/TaskProcessor.h>

#include "BrowserData.h"
#include "GenericMessageDialog.h"

namespace
{
    auto constexpr RefreshInterval = std::chrono::minutes(20);
}

void BrowserController::init()
{
    _refreshProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _data = _BrowserData::create();
}

void BrowserController::shutdown()
{
    // Releases the persister facade before it is shut down itself
    _refreshProcessor.reset();
    _data.reset();
}

void BrowserController::process()
{
    auto now = std::chrono::steady_clock::now();
    if (!_lastRefreshTime) {
        _lastRefreshTime = now;
    }
    if (now - *_lastRefreshTime >= RefreshInterval) {
        _lastRefreshTime = now;
        refresh(false);
    }

    _refreshProcessor->process();
    _data->processPendingRequests();
}

namespace
{
    bool isVisibleInWorkspace(NetworkResourceRawTO const& rawTO, WorkspaceType workspaceType, std::optional<std::string> const& userName)
    {
        switch (workspaceType) {
        case WorkspaceType_Private:
            return userName.has_value() && rawTO->userName == userName.value();
        case WorkspaceType_Public:
            return rawTO->workspaceType == WorkspaceType_Public || rawTO->workspaceType == WorkspaceType_AlienProject;
        case WorkspaceType_AlienProject:
            return rawTO->workspaceType == WorkspaceType_AlienProject;
        default:
            return false;
        }
    }
}

void BrowserController::refresh(bool withRetry)
{
    _refreshProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleGetNetworkResources(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = withRetry}, GetNetworkResourcesRequestData());
        },
        [&](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchGetNetworkResourcesData(requestId);
            _data->userTOs = data.userTOs;
            _data->ownEmojiTypeBySimId = data.emojiTypeByResourceId;

            auto userName = NetworkService::get().getLoggedInUserName();
            for (auto& [workspaceId, workspace] : _data->workspaces) {
                workspace.rawTOs.clear();
                for (auto const& rawTO : data.resourceTOs) {
                    if (rawTO->resourceType == workspaceId.resourceType && isVisibleInWorkspace(rawTO, workspaceId.workspaceType, userName)) {
                        workspace.rawTOs.emplace_back(rawTO);
                    }
                }
                _data->createTreeTOs(workspace);
            }
            _data->sortUserList();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

bool BrowserController::isRefreshing() const
{
    return _refreshProcessor->pendingTasks();
}

BrowserData const& BrowserController::getData() const
{
    return _data;
}
