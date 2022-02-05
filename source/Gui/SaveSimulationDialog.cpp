#include "SaveSimulationDialog.h"

#include <imgui.h>
#include <ImFileDialog.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"

_SaveSimulationDialog::_SaveSimulationDialog(SimulationController const& simController)
    : _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.save simulation.starting path", path.string());
}

_SaveSimulationDialog::~_SaveSimulationDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.save simulation.starting path", _startingPath);
}

void _SaveSimulationDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SimulationSaveDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        DeserializedSimulation sim;
        sim.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        sim.settings = _simController->getSettings();
        sim.symbolMap = _simController->getSymbolMap();
        sim.content = _simController->getClusteredSimulationData();

        if (!Serializer::serializeSimulationToFile(firstFilename.string(), sim)) {
            MessageDialog::getInstance().show("Save simulation", "The simulation could not be saved to the specified file.");
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _SaveSimulationDialog::show()
{
    ifd::FileDialog::Instance().Save("SimulationSaveDialog", "Save simulation", "Simulation file (*.sim){.sim},.*", _startingPath);
}
