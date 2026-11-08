#include "FileTransferController.h"

#include <boost/algorithm/string.hpp>

#include <Base/GlobalSettings.h>

#include <Engine/Interface/SimulationFacade.h>

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "RecorderController.h"
#include "SimulationLibraryController.h"

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
            SimulationLibraryController::get().onLoad(filename);
        });
}

void FileTransferController::onSaveSimulationDialog()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save simulation", "Simulation file (*.sim){.sim},.*", _referencePath, [&](std::filesystem::path const& filename) {
            auto filenameCopy = filename;
            _referencePath = filenameCopy.remove_filename().string();
            SimulationLibraryController::get().onSaveToFile(filename);
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
            if (!SerializerService::get().serializeGenomeToFile(firstFilename, genome)) {
                GenericMessageDialog::get().information("Save genome", "The selected file could not be saved.");
                return;
            }

            // The file may lie in the folder of the genomes of this computer
            SimulationLibraryController::get().refresh();
            if (afterSaveFunc) {
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
    _referencePath = GlobalSettings::get().getValue("dialogs.directory.reference path", _referencePath);
}

void FileTransferController::process() {}

void FileTransferController::shutdown()
{
    GlobalSettings::get().setValue("dialogs.directory.reference path", _referencePath);
}
