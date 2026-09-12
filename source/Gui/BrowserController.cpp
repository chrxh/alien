#include "BrowserController.h"

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

            for (auto& [workspaceId, workspace] : _data->workspaces) {
                workspace.rawTOs.clear();
                auto userName = NetworkService::get().getLoggedInUserName().value_or("");
                for (auto const& rawTO : data.resourceTOs) {
                    if (rawTO->resourceType == workspaceId.resourceType) {
                        if ((workspaceId.workspaceType == WorkspaceType_Private && rawTO->userName == userName)
                            || ((workspaceId.workspaceType == WorkspaceType_Public || workspaceId.workspaceType == WorkspaceType_AlienProject)
                                && rawTO->workspaceType == workspaceId.workspaceType)) {
                            workspace.rawTOs.emplace_back(rawTO);
                        }
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
