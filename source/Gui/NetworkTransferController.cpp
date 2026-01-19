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
#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/PersisterFacade.h>

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
                _PersisterFacade::get()->shutdown();
                _SimulationFacade::get()->closeSimulation();
                std::optional<std::string> errorMessage;
                auto const& deserializedSimulation = std::get<DeserializedSimulation>(data.resourceData);
                try {
                    _SimulationFacade::get()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                    _SimulationFacade::get()->setRealTime(deserializedSimulation.auxiliaryData.realTime);
                    _SimulationFacade::get()->setSimulationData(deserializedSimulation.mainData);
                    _SimulationFacade::get()->setStatisticsHistory(deserializedSimulation.statistics);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }
                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    _SimulationFacade::get()->closeSimulation();
                    _SimulationFacade::get()->newSimulation(
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.worldSize,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                }
                _PersisterFacade::get()->restart();

                Viewport::get().setCenterInWorldPos(deserializedSimulation.auxiliaryData.center);
                Viewport::get().setZoomFactor(deserializedSimulation.auxiliaryData.zoom);
                TemporalControlWindow::get().onSnapshot();

                printOverlayMessage(data.resourceName);
            } else {
                EditorController::get().setOn(true);
                GenomeEditorWindow::get().openTab(std::nullopt, std::get<GenomeDesc>(data.resourceData));
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
            return _PersisterFacade::get()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _PersisterFacade::get()->fetchUploadNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
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
        [&](auto const& requestId) {
            _PersisterFacade::get()->fetchReplaceNetworkResourcesData(requestId);
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
