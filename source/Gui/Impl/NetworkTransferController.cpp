#include "NetworkTransferController.h"

#include <Fonts/IconsFontAwesome5.h>

#include <Base/VersionParserService.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/TaskProcessor.h>

#include "BrowserWindow.h"
#include "EditorController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "NewSimulationService.h"
#include "OverlayController.h"
#include <Engine/Interface/SimulationFacade.h>
#include <Persister/Interface/PersisterFacade.h>

void NetworkTransferController::init()
{

    _downloadProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _uploadProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _replaceProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _deleteProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _editProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _moveProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
}

void NetworkTransferController::onDownload(DownloadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Downloading ...");

    _downloadProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchDownloadNetworkResourcesData(requestId);

            if (data.resourceType == NetworkResourceType_Simulation) {
                if (auto errorMessage = NewSimulationService::get().loadSimulation(std::get<SimulationDesc>(data.resourceData))) {
                    showMessage("Error", *errorMessage);
                }
                printOverlayMessage(data.resourceName);
            } else {
                EditorController::get().setOn(true);
                GenomeEditorWindow::get().openTab(std::get<GenomeDesc>(data.resourceData), false);
            }
            if (VersionParserService::get().isVersionNewer(data.resourceVersion)) {
                std::string dataTypeString = data.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
                GenericMessageDialog::get().markdownInformation(
                    "Warning",
                    "The download was successful but the " + dataTypeString + " was generated using a more recent version of ALIEN. Consequently, the "
                        + dataTypeString
                        + " might not function as expected.\n\nPlease visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
            }
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

namespace
{
    void showUploadConfirmation(NetworkResourceType resourceType, WorkspaceType workspaceType, std::string const& resourceName, std::string const& resourceId)
    {
        auto isShared = workspaceType == WorkspaceType_Public;
        auto message = isShared ? "\"" + resourceName + "\" is now visible in the Community workspace. It has also been posted on the ALIEN Discord."
                                : "\"" + resourceName + "\" is now in your Private workspace. Only you can see it.";
        GenericMessageDialog::get().informationWithAction(isShared ? "Shared" : "Uploaded", message, ICON_FA_EYE "  Show in browser", [=] {
            BrowserWindow::get().showResource(resourceType, workspaceType, resourceId);
        });
    }
}

void NetworkTransferController::onUpload(UploadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Uploading ...");

    auto resourceType = std::holds_alternative<UploadNetworkResourceRequestData::SimulationData>(requestData.data) ? NetworkResourceType_Simulation
                                                                                                                   : NetworkResourceType_Genome;
    _uploadProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [resourceType, workspaceType = requestData.workspaceType, resourceName = requestData.resourceWithoutFolderName](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchUploadNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
            showUploadConfirmation(resourceType, workspaceType, resourceName, data.resourceId);
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onReplace(ReplaceNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Replacing ...");

    _replaceProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleReplaceNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&, resourceId = requestData.resourceId](auto const& requestId) {
            _PersisterFacade::get()->fetchReplaceNetworkResourcesData(requestId);
            BrowserWindow::get().onPreviewPictureChanged(resourceId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onDelete(DeleteNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Deleting ...");

    _deleteProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleDeleteNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _PersisterFacade::get()->fetchDeleteNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onEdit(EditNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Applying changes ...");

    _editProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleEditNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _PersisterFacade::get()->fetchEditNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onMove(MoveNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Changing visibility ...");

    _moveProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleMoveNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _PersisterFacade::get()->fetchMoveNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::process()
{
    _downloadProcessor->process();
    _uploadProcessor->process();
    _replaceProcessor->process();
    _deleteProcessor->process();
    _editProcessor->process();
    _moveProcessor->process();
}
