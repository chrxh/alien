#include "FileTransferController.h"

#include <ImFileDialog.h>

#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/TaskProcessor.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"

namespace
{
    auto constexpr FileTransferSenderId = "FileTransfer";
}

void FileTransferController::init(PersisterFacade persisterFacade, SimulationFacade simulationFacade)
{
    _persisterFacade = persisterFacade;
    _simulationFacade = simulationFacade;
    _openSimulationProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _saveSimulationProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
}

void FileTransferController::onOpenSimulation()
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();

            printOverlayMessage("Loading ...");

        _openSimulationProcessor->executeTask(
            [&](auto const& senderId) {
                auto senderInfo = SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true};
                auto readData = ReadSimulationRequestData{firstFilename.string()};
                return _persisterFacade->scheduleReadSimulation(senderInfo, readData);
            },
            [&](auto const& requestId) {
                auto const& data = _persisterFacade->fetchReadSimulationData(requestId);
                _persisterFacade->shutdown();

                _simulationFacade->closeSimulation();

                std::optional<std::string> errorMessage;
                try {
                    _simulationFacade->newSimulation(
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
                        data.deserializedSimulation.auxiliaryData.timestep,
                        data.deserializedSimulation.auxiliaryData.generalSettings,
                        data.deserializedSimulation.auxiliaryData.simulationParameters);
                }
                _persisterFacade->restart();

                Viewport::get().setCenterInWorldPos(data.deserializedSimulation.auxiliaryData.center);
                Viewport::get().setZoomFactor(data.deserializedSimulation.auxiliaryData.zoom);
                TemporalControlWindow::get().onSnapshot();
                printOverlayMessage(data.filename.string());
            },
            [](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
    });
}

void FileTransferController::onSaveSimulation()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();
            printOverlayMessage("Saving ...");
            _saveSimulationProcessor->executeTask(
                [&, firstFilename = firstFilename](auto const& senderId) {
                    auto senderInfo = SenderInfo{.senderId = senderId, .wishResultData = false, .wishErrorInfo = true};
                    auto readData = SaveSimulationRequestData{firstFilename.string(), Viewport::get().getZoomFactor(), Viewport::get().getCenterInWorldPos()};
                    return _persisterFacade->scheduleSaveSimulation(senderInfo, readData);
                },
                [](auto const&) { },
                [](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
        });
}

void FileTransferController::process()
{
    _openSimulationProcessor->process();
    _saveSimulationProcessor->process();
}
