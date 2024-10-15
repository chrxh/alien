#include "NetworkTransferController.h"

#include "Base/VersionChecker.h"
#include "EngineInterface/SimulationController.h"
#include "PersisterInterface/TaskProcessor.h"

#include "MessageDialog.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"
#include "EditorController.h"
#include "GenomeEditorWindow.h"
#include "BrowserWindow.h"

void NetworkTransferController::init(
    SimulationController const& simController,
    PersisterController const& persisterController,
    TemporalControlWindow const& temporalControlWindow,
    EditorController const& editorController,
    BrowserWindow const& browserWindow)
{
    _simController = simController;
    _persisterController = persisterController;
    _temporalControlWindow = temporalControlWindow;
    _editorController = editorController;
    _browserWindow = browserWindow;
    _downloadProcessor = _TaskProcessor::createTaskProcessor(_persisterController);
    _uploadProcessor = _TaskProcessor::createTaskProcessor(_persisterController);
}

void NetworkTransferController::onDownload(DownloadNetworkResourceRequestData const& requestData)
{
    _downloadProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterController->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            auto data = _persisterController->fetchDownloadNetworkResourcesData(requestId);

            if (data.resourceType == NetworkResourceType_Simulation) {
                _persisterController->shutdown();
                _simController->closeSimulation();
                std::optional<std::string> errorMessage;
                auto const& deserializedSimulation = std::get<DeserializedSimulation>(data.resourceData);
                try {
                    _simController->newSimulation(
                        data.resourceName,
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.generalSettings,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                    _simController->setRealTime(deserializedSimulation.auxiliaryData.realTime);
                    _simController->setClusteredSimulationData(deserializedSimulation.mainData);
                    _simController->setStatisticsHistory(deserializedSimulation.statistics);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }
                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    _simController->closeSimulation();
                    _simController->newSimulation(
                        data.resourceName,
                        deserializedSimulation.auxiliaryData.timestep,
                        deserializedSimulation.auxiliaryData.generalSettings,
                        deserializedSimulation.auxiliaryData.simulationParameters);
                }
                _persisterController->restart();

                Viewport::setCenterInWorldPos(deserializedSimulation.auxiliaryData.center);
                Viewport::setZoomFactor(deserializedSimulation.auxiliaryData.zoom);
                _temporalControlWindow->onSnapshot();
            } else {
                _editorController->setOn(true);
                _editorController->getGenomeEditorWindow()->openTab(std::get<GenomeDescription>(data.resourceData));
            }
            if (VersionChecker::isVersionNewer(data.resourceVersion)) {
                std::string dataTypeString = data.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
                MessageDialog::get().information(
                    "Warning",
                    "The download was successful but the " + dataTypeString
                        + " was generated using a more recent\n"
                          "version of ALIEN. Consequently, the "
                        + dataTypeString
                        + "might not function as expected.\n"
                          "Please visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
            }
        },
        [](auto const& errors) { MessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::onUpload(UploadNetworkResourceRequestData const& requestData)
{
    _uploadProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterController->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [&](auto const& requestId) {
            _persisterController->fetchUploadNetworkResourcesData(requestId);
            _browserWindow->onRefresh();
        },
        [](auto const& errors) { MessageDialog::get().information("Error", errors); });
}

void NetworkTransferController::process()
{
    _downloadProcessor->process();
    _uploadProcessor->process();
}
