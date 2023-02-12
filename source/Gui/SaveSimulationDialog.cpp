#include "SaveSimulationDialog.h"

#include <imgui.h>
#include <ImFileDialog.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "Viewport.h"

_SaveSimulationDialog::_SaveSimulationDialog(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
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
        sim.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        sim.auxiliaryData.zoom = _viewport->getZoomFactor();
        sim.auxiliaryData.center = _viewport->getCenterInWorldPos();
        sim.auxiliaryData.generalSettings = _simController->getGeneralSettings();
        sim.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
        sim.mainData = _simController->getClusteredSimulationData();

        if (!Serializer::serializeSimulationToFiles(firstFilename.string(), sim)) {
            MessageDialog::getInstance().show("Save simulation", "The simulation could not be saved to the specified file.");
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _SaveSimulationDialog::show()
{
    ifd::FileDialog::Instance().Save("SimulationSaveDialog", "Save simulation", "Simulation file (*.sim){.sim},.*", _startingPath);
}
