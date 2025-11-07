#include "FileTransferController.h"

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/TaskProcessor.h>

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"

#include <ImFileDialog.h>
#include "Provider.h"

void FileTransferController::onOpenSimulationDialog()
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& filename) {
            auto filenameCopy = filename;
            _referencePath = filenameCopy.remove_filename().string();
            onOpenSimulation(filename);
        });
}

void FileTransferController::onOpenSimulation(std::filesystem::path const& filename)
{
    printOverlayMessage("Loading ...");

    _openSimulationProcessor->executeTask(
        [&](auto const& senderId) {
            auto senderInfo = SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true};
            auto readData = ReadSimulationRequestData{.filename = filename.string(), .initSimulation = false};
            return Provider::getPersisterFacade()->scheduleReadSimulation(senderInfo, readData);
        },
        [&](auto const& requestId) {
            auto const& data = Provider::getPersisterFacade()->fetchReadSimulationData(requestId);
            Provider::getPersisterFacade()->shutdown();

            Provider::getSimulationFacade()->closeSimulation();

            std::optional<std::string> errorMessage;
            try {
                Provider::getSimulationFacade()->newSimulation(
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.worldSize,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
                Provider::getSimulationFacade()->setSimulationData(data.deserializedSimulation.mainData);
                Provider::getSimulationFacade()->setStatisticsHistory(data.deserializedSimulation.statistics);
                Provider::getSimulationFacade()->setRealTime(data.deserializedSimulation.auxiliaryData.realTime);
            } catch (CudaMemoryAllocationException const& exception) {
                errorMessage = exception.what();
            } catch (...) {
                errorMessage = "Failed to load simulation.";
            }

            if (errorMessage) {
                showMessage("Error", *errorMessage);
                Provider::getSimulationFacade()->closeSimulation();
                Provider::getSimulationFacade()->newSimulation(
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.worldSize,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
            }
            Provider::getPersisterFacade()->restart();

            Viewport::get().setCenterInWorldPos(data.deserializedSimulation.auxiliaryData.center);
            Viewport::get().setZoomFactor(data.deserializedSimulation.auxiliaryData.zoom);
            TemporalControlWindow::get().onSnapshot();
            printOverlayMessage(data.filename.string());
        },
        [](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
}

void FileTransferController::onSaveSimulationDialog()
{
    GenericFileDialog::get().showSaveFileDialog("Save simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _referencePath = firstFilenameCopy.remove_filename().string();
        printOverlayMessage("Saving ...");
        _saveSimulationProcessor->executeTask(
            [&, firstFilename = firstFilename](auto const& senderId) {
                auto senderInfo = SenderInfo{.senderId = senderId, .wishResultData = false, .wishErrorInfo = true};
                auto readData = SaveSimulationRequestData{firstFilename.string(), Viewport::get().getZoomFactor(), Viewport::get().getCenterInWorldPos()};
                return Provider::getPersisterFacade()->scheduleSaveSimulation(senderInfo, readData);
            },
            [](auto const&) {},
            [](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
    });
}

void FileTransferController::init()
{

    _openSimulationProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());
    _saveSimulationProcessor = _TaskProcessor::createTaskProcessor(Provider::getPersisterFacade());

    _referencePath = GlobalSettings::get().getValue("dialogs.directory.reference path", _referencePath);
}

void FileTransferController::process()
{
    _openSimulationProcessor->process();
    _saveSimulationProcessor->process();
}

void FileTransferController::shutdown()
{
    GlobalSettings::get().setValue("dialogs.directory.reference path", _referencePath);
}
