#include "SaveSimulationDialog.h"

#include <imgui.h>

#include "EngineImpl/SimulationController.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"

_SaveSimulationDialog::_SaveSimulationDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _SaveSimulationDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SimulationSaveDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        DeserializedSimulation sim;
        sim.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
        sim.settings = _simController->getSettings();
        sim.symbolMap = _simController->getSymbolMap();
        sim.content = _simController->getSimulationData({0, 0}, _simController->getWorldSize());

        Serializer serializer = boost::make_shared<_Serializer>();
        serializer->serializeSimulationToFile(firstFilename.string(), sim);
    }
    ifd::FileDialog::Instance().Close();
}

void _SaveSimulationDialog::show()
{
    ifd::FileDialog::Instance().Save("SimulationSaveDialog", "Save simulation", "Simulation file (*.sim){.sim},.*");
}
