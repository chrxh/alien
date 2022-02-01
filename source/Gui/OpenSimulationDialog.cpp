#include "OpenSimulationDialog.h"

#include <imgui.h>
#include "ImFileDialog.h"

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "TemporalControlWindow.h"

_OpenSimulationDialog::_OpenSimulationDialog(
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow,
    StatisticsWindow const& statisticsWindow,
    Viewport const& viewport)
    : _simController(simController)
    , _temporalControlWindow(temporalControlWindow)
    , _statisticsWindow(statisticsWindow)
    , _viewport(viewport)
{}

void _OpenSimulationDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SimulationOpenDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        Serializer serializer = std::make_shared<_Serializer>();

        DeserializedSimulation deserializedData;
        if (serializer->deserializeSimulationFromFile(firstFilename.string(), deserializedData)) {
            _simController->closeSimulation();
            _statisticsWindow->reset();

            _simController->newSimulation(
                deserializedData.timestep, deserializedData.settings, deserializedData.symbolMap);
            _simController->setClusteredSimulationData(deserializedData.content);
            _viewport->setCenterInWorldPos(
                {toFloat(deserializedData.settings.generalSettings.worldSizeX) / 2,
                 toFloat(deserializedData.settings.generalSettings.worldSizeY) / 2});
            _viewport->setZoomFactor(2.0f);
            _temporalControlWindow->onSnapshot();
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenSimulationDialog::show()
{
    ifd::FileDialog::Instance().Open(
        "SimulationOpenDialog", "Open simulation", "Simulation file (*.sim){.sim},.*", false);
}
