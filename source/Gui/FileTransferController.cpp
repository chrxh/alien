#include "FileTransferController.h"

#include <ImFileDialog.h>

#include "EngineInterface/SimulationController.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"
#include "OverlayMessageController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"

namespace
{
    auto constexpr FileTransferSenderId = "FileTransfer";
}

FileTransferController& FileTransferController::get()
{
    static FileTransferController instance;
    return instance;
}

void FileTransferController::init(
    PersisterController const& persisterController,
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow)
{
    _persisterController = persisterController;
    _simController = simController;
    _temporalControlWindow = temporalControlWindow;
}

void FileTransferController::onOpenSimulation()
{
    GenericFileDialogs::get().showOpenFileDialog(
        "Open simulation", "Simulation file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            printOverlayMessage("Loading ...");

            auto senderInfo = SenderInfo{.senderId = SenderId{FileTransferSenderId}, .wishResultData = true, .wishErrorInfo = true};
            auto readData = ReadSimulationRequestData{firstFilename.string()};
            _readSimulationRequestIds.emplace_back(_persisterController->scheduleReadSimulationFromFile(senderInfo, readData));
        });
}

void FileTransferController::onSaveSimulation()
{
    GenericFileDialogs::get().showSaveFileDialog(
        "Save simulation", "Simulation file (*.sim){.sim},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();
            printOverlayMessage("Saving ...");
            auto senderInfo = SenderInfo{.senderId = SenderId{FileTransferSenderId}, .wishResultData = false, .wishErrorInfo = true};
            auto saveData = SaveSimulationRequestData{firstFilename.string(), Viewport::getZoomFactor(), Viewport::getCenterInWorldPos()};
            _persisterController->scheduleSaveSimulationToFile(senderInfo, saveData);
        });
}

void FileTransferController::process()
{
    std::vector<PersisterRequestId> newReadSimulationRequestIds;
    for (auto const& requestId : _readSimulationRequestIds) {
        auto state = _persisterController->getRequestState(requestId);
        if (state == PersisterRequestState::Finished) {
            auto const& data = _persisterController->fetchReadSimulationData(requestId);
            _persisterController->shutdown();

            _simController->closeSimulation();

            std::optional<std::string> errorMessage;
            try {
                _simController->newSimulation(
                    data.simulationName,
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.generalSettings,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
                _simController->setClusteredSimulationData(data.deserializedSimulation.mainData);
                _simController->setStatisticsHistory(data.deserializedSimulation.statistics);
                _simController->setRealTime(data.deserializedSimulation.auxiliaryData.realTime);
            } catch (CudaMemoryAllocationException const& exception) {
                errorMessage = exception.what();
            } catch (...) {
                errorMessage = "Failed to load simulation.";
            }

            if (errorMessage) {
                showMessage("Error", *errorMessage);
                _simController->closeSimulation();
                _simController->newSimulation(
                    std::nullopt,
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.generalSettings,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
            }
            _persisterController->restart();

            Viewport::setCenterInWorldPos(data.deserializedSimulation.auxiliaryData.center);
            Viewport::setZoomFactor(data.deserializedSimulation.auxiliaryData.zoom);
            _temporalControlWindow->onSnapshot();
            printOverlayMessage(data.simulationName + ".sim");
        }
        if (state == PersisterRequestState::InQueue || state == PersisterRequestState::InProgress) {
            newReadSimulationRequestIds.emplace_back(requestId);
        }
    }
    _readSimulationRequestIds = newReadSimulationRequestIds;

    auto criticalErrors = _persisterController->fetchAllErrorInfos(SenderId{FileTransferSenderId});
    if (!criticalErrors.empty()) {
        MessageDialog::get().information("Error", criticalErrors);
    }
}
