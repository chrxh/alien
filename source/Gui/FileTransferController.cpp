#include "FileTransferController.h"

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/TaskProcessor.h>

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "TemporalControlWindow.h"
#include "Viewport.h"

#include <ImFileDialog.h>
#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/PersisterFacade.h>

#include "PersisterInterface/SerializerService.h"

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
            return _PersisterFacade::get()->scheduleReadSimulation(senderInfo, readData);
        },
        [&](auto const& requestId) {
            auto const& data = _PersisterFacade::get()->fetchReadSimulationData(requestId);
            _PersisterFacade::get()->shutdown();

            _SimulationFacade::get()->closeSimulation();

            std::optional<std::string> errorMessage;
            try {
                _SimulationFacade::get()->newSimulation(
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.worldSize,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
                _SimulationFacade::get()->setSimulationData(data.deserializedSimulation.mainData);
                _SimulationFacade::get()->setStatisticsHistory(data.deserializedSimulation.statistics);
                _SimulationFacade::get()->setRealTime(data.deserializedSimulation.auxiliaryData.realTime);
            } catch (CudaMemoryAllocationException const& exception) {
                errorMessage = exception.what();
            } catch (...) {
                errorMessage = "Failed to load simulation.";
            }

            if (errorMessage) {
                showMessage("Error", *errorMessage);
                _SimulationFacade::get()->closeSimulation();
                _SimulationFacade::get()->newSimulation(
                    data.deserializedSimulation.auxiliaryData.timestep,
                    data.deserializedSimulation.auxiliaryData.worldSize,
                    data.deserializedSimulation.auxiliaryData.simulationParameters);
            }
            _PersisterFacade::get()->restart();

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
                return _PersisterFacade::get()->scheduleSaveSimulation(senderInfo, readData);
            },
            [](auto const&) {},
            [](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
    });
}

void FileTransferController::onOpenGenomeDialog(std::function<void(GenomeDesc const&)> const& openFunc)
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open genome",
        "Genome (*.genome){.genome},.*",
        _referencePath,
        [&, openFunc = openFunc](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();
            GenomeDesc genome;
            if (!SerializerService::get().deserializeGenomeFromFile(genome, firstFilename.string())) {
                GenericMessageDialog::get().information("Open genome", "The selected file could not be opened.");
            } else {
                openFunc(genome);
            }
        });
}

void FileTransferController::onSaveGenomeDialog(GenomeDesc const& genome, std::function<void()> const& saveFunc)
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save genome", "Genome (*.genome){.genome},.*", _referencePath, [&, genome = genome, saveFunc = saveFunc](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();
            if (!SerializerService::get().serializeGenomeToFile(firstFilename.string(), genome)) {
                GenericMessageDialog::get().information("Save genome", "The selected file could not be saved.");
            } else if (saveFunc) {
                saveFunc();
            }
        });
}

void FileTransferController::init()
{

    _openSimulationProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
    _saveSimulationProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());

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
