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
#include "SimulationFacadeProvider.h"
#include "PersisterFacadeProvider.h"

void NetworkTransferController::init()
{

    _downloadProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
    _uploadProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
    _replaceProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
    _deleteProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
    _editProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
    _moveProcessor = _TaskProcessor::createTaskProcessor(PersisterFacadeProvider::getPersisterFacade());
}

void NetworkTransferController::onDownload(DownloadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Downloading ...");

    _downloadProcessor->executeTask(
        [&](auto const& senderId) {
            return PersisterFacadeProvider::getPersisterFacade()->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            auto data = PersisterFacadeProvider::getPersisterFacade()->fetchDownloadNetworkResourcesData(requestId);

            if (data.resourceType == NetworkResourceType_Simulation) {
                PersisterFacadeProvider::getPersisterFacade()->shutdown();
                SimulationFacadeProvider::getSimulationFacade()->closeSimulation();
                std::optional<std::string> errorMessage;
                auto const& deserializedSimulation = std::get<DeserializedSimulation>(data.resourceData);
                try {
                    SimulationFacadeProvider::getSimulationFacade()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                    SimulationFacadeProvider::getSimulationFacade()->setRealTime(deserializedSimulation.auxiliaryData.realTime);
                    SimulationFacadeProvider::getSimulationFacade()->setSimulationData(deserializedSimulation.mainData);
                    SimulationFacadeProvider::getSimulationFacade()->setStatisticsHistory(deserializedSimulation.statistics);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }
                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    SimulationFacadeProvider::getSimulationFacade()->closeSimulation();
                    SimulationFacadeProvider::getSimulationFacade()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                }
                PersisterFacadeProvider::getPersisterFacade()->restart();

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
            return PersisterFacadeProvider::getPersisterFacade()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            PersisterFacadeProvider::getPersisterFacade()->fetchUploadNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onReplace(ReplaceNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Replacing ...");

    _replaceProcessor->executeTask(
        [&](auto const& senderId) {
            return PersisterFacadeProvider::getPersisterFacade()->scheduleReplaceNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            PersisterFacadeProvider::getPersisterFacade()->fetchReplaceNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onDelete(DeleteNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Deleting ...");

    _deleteProcessor->executeTask(
        [&](auto const& senderId) {
            return PersisterFacadeProvider::getPersisterFacade()->scheduleDeleteNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            PersisterFacadeProvider::getPersisterFacade()->fetchDeleteNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onEdit(EditNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Applying changes ...");

    _editProcessor->executeTask(
        [&](auto const& senderId) {
            return PersisterFacadeProvider::getPersisterFacade()->scheduleEditNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            PersisterFacadeProvider::getPersisterFacade()->fetchEditNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onMove(MoveNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Changing visibility ...");

    _moveProcessor->executeTask(
        [&](auto const& senderId) {
            return PersisterFacadeProvider::getPersisterFacade()->scheduleMoveNetworkResource(SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            PersisterFacadeProvider::getPersisterFacade()->fetchMoveNetworkResourcesData(requestId);
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
