#include "FileTransferController.h"

#include <boost/algorithm/string.hpp>

#include <Base/GlobalSettings.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/TaskProcessor.h>

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "NewSimulationService.h"
#include "OverlayController.h"
#include "RecorderController.h"
#include "Viewport.h"

#include <ImFileDialog.h>
#include <Engine/Interface/SimulationFacade.h>
#include <Persister/Interface/PersisterFacade.h>

#include "Persister/Interface/SerializerService.h"

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
            if (auto errorMessage = NewSimulationService::get().loadSimulation(data.simulationDesc)) {
                showMessage("Error", *errorMessage);
            }
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

void FileTransferController::onSaveGenomeDialog(GenomeDesc const& genome, std::function<void()> const& afterSaveFunc)
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save genome", "Genome (*.genome){.genome},.*", _referencePath, [&, genome = genome, afterSaveFunc = afterSaveFunc](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _referencePath = firstFilenameCopy.remove_filename().string();
            if (!SerializerService::get().serializeGenomeToFile(firstFilename.string(), genome)) {
                GenericMessageDialog::get().information("Save genome", "The selected file could not be saved.");
            } else if (afterSaveFunc) {
                afterSaveFunc();
            }
        });
}

void FileTransferController::onOpenCameraFlightDialog(std::function<void()> const& afterOpenFunc)
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open camera flight",
        "Camera flight (*.flight.json){.flight.json,.json},.*",
        _referencePath,
        [&, afterOpenFunc = afterOpenFunc](std::filesystem::path const& path) {
            _referencePath = std::filesystem::path(path).remove_filename().string();
            if (!RecorderController::get().loadFlight(path)) {
                GenericMessageDialog::get().information("Open camera flight", "The selected file could not be opened.");
                return;
            }
            afterOpenFunc();
            printOverlayMessage("Camera flight loaded from " + path.filename().string());
        });
}

void FileTransferController::onSaveCameraFlightDialog()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save camera flight", "Camera flight (*.flight.json){.flight.json,.json},.*", _referencePath, [&](std::filesystem::path const& path) {
            // Camera flights are always written as JSON files, so that a file selected by mistake, e.g. a simulation, is not overwritten
            auto filename = path;
            if (!boost::iequals(filename.extension().string(), ".json")) {
                filename += ".flight.json";
            }
            _referencePath = std::filesystem::path(filename).remove_filename().string();
            if (!RecorderController::get().saveFlight(filename)) {
                GenericMessageDialog::get().information("Save camera flight", "The camera flight could not be saved.");
                return;
            }
            printOverlayMessage("Camera flight saved as " + filename.filename().string());
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
