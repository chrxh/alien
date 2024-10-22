#include "NetworkTransferController.h"

#include "Base/VersionChecker.h"
#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/TaskProcessor.h"

#include "GenericMessageDialog.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"
#include "EditorController.h"
#include "GenomeEditorWindow.h"
#include "BrowserWindow.h"
#include "OverlayMessageController.h"

void NetworkTransferController::init(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
    _downloadProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _uploadProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _replaceProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _deleteProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);

    MainLoopEntityController::get().registerObject(this);
}

void NetworkTransferController::onDownload(DownloadNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Downloading ...");

    _downloadProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterFacade->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            auto data = _persisterFacade->fetchDownloadNetworkResourcesData(requestId);

            if (data.resourceType == NetworkResourceType_Simulation) {
                _persisterFacade->shutdown();
                _simulationFacade->closeSimulation();
                std::optional<std::string> errorMessage;
                auto const& deserializedSimulation = std::get<DeserializedSimulation>(data.resourceData);
                try {
                    _simulationFacade->newSimulation(
                        data.resourceName,
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.generalSettings,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                    _simulationFacade->setRealTime(deserializedSimulation.auxiliaryData.realTime);
                    _simulationFacade->setClusteredSimulationData(deserializedSimulation.mainData);
                    _simulationFacade->setStatisticsHistory(deserializedSimulation.statistics);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }
                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    _simulationFacade->closeSimulation();
                    _simulationFacade->newSimulation(
                        data.resourceName,
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.generalSettings,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                }
                _persisterFacade->restart();

                Viewport::get().setCenterInWorldPos(deserializedSimulation.auxiliaryData.center);
                Viewport::get().setZoomFactor(deserializedSimulation.auxiliaryData.zoom);
                TemporalControlWindow::get().onSnapshot();
            } else {
                EditorController::get().setOn(true);
                GenomeEditorWindow::get().openTab(std::get<GenomeDescription>(data.resourceData));
            }
            if (VersionChecker::isVersionNewer(data.resourceVersion)) {
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
            return _persisterFacade->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _persisterFacade->fetchUploadNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onReplace(ReplaceNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Replacing ...");

    _replaceProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterFacade->scheduleReplaceNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _persisterFacade->fetchReplaceNetworkResourcesData(requestId);
            BrowserWindow::get().onRefresh();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onDelete(DeleteNetworkResourceRequestData const& requestData)
{
    printOverlayMessage("Deleting ...");

    _deleteProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterFacade->scheduleDeleteNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _persisterFacade->fetchDeleteNetworkResourcesData(requestId);
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
}
