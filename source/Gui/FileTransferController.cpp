#include "FileTransferController.h"

#include <ImFileDialog.h>

#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/TaskProcessor.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"
#include "OverlayMessageController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"

namespace
{
    auto constexpr FileTransferSenderId = "FileTransfer";
}

void FileTransferController::init(
    PersisterFacade const& persisterFacade,
    SimulationFacade const& simulationFacade,
    TemporalControlWindow const& temporalControlWindow)
{
    _persisterFacade = persisterFacade;
    _simulationFacade = simulationFacade;
    _temporalControlWindow = temporalControlWindow;
    _openSimulationProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
}

void FileTransferController::onOpenSimulation()
{
    GenericFileDialogs::get().showOpenFileDialog(
        "Open simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();

            printOverlayMessage("Loading ...");

        _openSimulationProcessor->executeTask(
            [&](auto const& senderId) {
                auto senderInfo = SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true};
                auto readData = ReadSimulationRequestData{firstFilename.string()};
                return _persisterFacade->scheduleReadSimulationFromFile(senderInfo, readData);
            },
            [&](auto const& requestId) {
                auto const& data = _persisterFacade->fetchReadSimulationData(requestId);
                _persisterFacade->shutdown();

                _simulationFacade->closeSimulation();

                std::optional<std::string> errorMessage;
                try {
                    _simulationFacade->newSimulation(
                        data.simulationName,
                        data.deserializedSimulation.auxiliaryData.timestep,
                        data.deserializedSimulation.auxiliaryData.generalSettings,
                        data.deserializedSimulation.auxiliaryData.simulationParameters);
                    _simulationFacade->setClusteredSimulationData(data.deserializedSimulation.mainData);
                    _simulationFacade->setStatisticsHistory(data.deserializedSimulation.statistics);
                    _simulationFacade->setRealTime(data.deserializedSimulation.auxiliaryData.realTime);
                } catch (CudaMemoryAllocationException const& exception) {
                    errorMessage = exception.what();
                } catch (...) {
                    errorMessage = "Failed to load simulation.";
                }

                if (errorMessage) {
                    showMessage("Error", *errorMessage);
                    _simulationFacade->closeSimulation();
                    _simulationFacade->newSimulation(
                        std::nullopt,
                        data.deserializedSimulation.auxiliaryData.timestep,
                        data.deserializedSimulation.auxiliaryData.generalSettings,
                        data.deserializedSimulation.auxiliaryData.simulationParameters);
                }
                _persisterFacade->restart();

                Viewport::get().setCenterInWorldPos(data.deserializedSimulation.auxiliaryData.center);
                Viewport::get().setZoomFactor(data.deserializedSimulation.auxiliaryData.zoom);
                _temporalControlWindow->onSnapshot();
                printOverlayMessage(data.simulationName + ".sim");
            },
            [](auto const& criticalErrors) { MessageDialog::get().information("Error", criticalErrors); });
    });
}

void FileTransferController::onSaveSimulation()
{
    GenericFileDialogs::get().showSaveFileDialog(
        "Save simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();
            printOverlayMessage("Saving ...");
            auto senderInfo = SenderInfo{.senderId = SenderId{FileTransferSenderId}, .wishResultData = false, .wishErrorInfo = true};
            auto saveData = SaveSimulationRequestData{firstFilename.string(), Viewport::get().getZoomFactor(), Viewport::get().getCenterInWorldPos()};
            _persisterFacade->scheduleSaveSimulationToFile(senderInfo, saveData);
        });
}

void FileTransferController::process()
{
    _openSimulationProcessor->process();
}
