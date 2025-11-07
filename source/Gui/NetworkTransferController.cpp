#include "NetworkTransferController.h"

#include <Base/VersionParserService.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/TaskProcessor.h>

#include "BrowserWindow.h"
#include "EditorController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "OverlayController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"
#include "Provider.h"

void NetworkTransferController::init()
{

    _downloadProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _uploadProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _replaceProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _deleteProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _editProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _moveProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
}

void NetworkTransferController::onDownload(DownloadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Downloading ...");

    _downloadProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            auto data = Provider::getPersisterFacade()->fetchDownloadNetworkResourcesData(requestId);

            if (data.resourceType == NetworkResourceType_Simulation) {
                Provider::getPersisterFacade()->shutdown();
                Provider::getSimulationFacade()->closeSimulation();
                std::optional<std::string> errorMessage;
                auto const& deserializedSimulation = std::get<DeserializedSimulation>(data.resourceData);
                try {
                    Provider::getSimulationFacade()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                    Provider::getSimulationFacade()->setRealTime(deserializedSimulation.auxiliaryData.realTime);
                    Provider::getSimulationFacade()->setSimulationData(deserializedSimulation.mainData);
                    Provider::getSimulationFacade()->setStatisticsHistory(deserializedSimulation.statistics);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }
                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    Provider::getSimulationFacade()->closeSimulation();
                    Provider::getSimulationFacade()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                }
                Provider::getPersisterFacade()->restart();

                Viewport::get().setCenterInWorldPos(deserializedSimulation.auxiliaryData.center);
                Viewport::get().setZoomFactor(deserializedSimulation.auxiliaryData.zoom);
                TemporalControlWindow::get().onSnapshot();

                printOverlayMessage(data.resourceName);
            } else {
                EditorController::get().setOn(true);
                GenomeEditorWindow::get().openTab(std::nullopt, std::get<GenomeDescription>(data.resourceData));
            }
            if (VersionParserService::get().isVersionNewer(data.resourceVersion)) {
                std::string dataTypeString = data.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
                GenericMessageDialog::get().information(
                    "Warning",
                    "The download was successful but the " + dataTypeString
                        + " was generated using a more recent\n"
                          "version of ALIEN. Consequently, the "
                        + dataTypeString
                        + "might not function as expected.\n"
                          "Please visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
            }
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onUpload(UploadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Uploading ...");

    _uploadProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            Provider::getPersisterFacade()->fetchUploadNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onReplace(ReplaceNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Replacing ...");

    _replaceProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleReplaceNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            Provider::getPersisterFacade()->fetchReplaceNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onDelete(DeleteNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Deleting ...");

    _deleteProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleDeleteNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            Provider::getPersisterFacade()->fetchDeleteNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onEdit(EditNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Applying changes ...");

    _editProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleEditNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            Provider::getPersisterFacade()->fetchEditNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onMove(MoveNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Changing visibility ...");

    _moveProcessor->executeTask(
        [&](auto const& senderId) {
            return Provider::getPersisterFacade()->scheduleMoveNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            Provider::getPersisterFacade()->fetchMoveNetworkResourcesData(requestId);
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
