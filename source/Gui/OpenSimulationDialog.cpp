#include "OpenSimulationDialog.h"

#include <imgui.h>
#include "ImFileDialog.h"

#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationController.h"
#include "StatisticsWindow.h"
#include "Viewport.h"

_OpenSimulationDialog::_OpenSimulationDialog(
    SimulationController const& simController,
    StatisticsWindow const& statisticsWindow,
    Viewport const& viewport)
    : _simController(simController)
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
        _simController->closeSimulation();

        _statisticsWindow->reset();

        Serializer serializer = boost::make_shared<_Serializer>();

        DeserializedSimulation deserializedData;
        serializer->deserializeSimulationFromFile(firstFilename.string(), deserializedData);

        _simController->newSimulation(deserializedData.timestep, deserializedData.settings, deserializedData.symbolMap);
        _simController->setSimulationData(deserializedData.content);
        _viewport->setCenterInWorldPos(
            {toFloat(deserializedData.settings.generalSettings.worldSizeX) / 2,
             toFloat(deserializedData.settings.generalSettings.worldSizeY) / 2});
        _viewport->setZoomFactor(2.0f);

/*
        Serializer serializer = boost::make_shared<_Serializer>();
        SerializedSimulation serializedData;
        serializer->loadSimulationDataFromFile(firstFilename.string(), serializedData);
        auto deserializedData = serializer->deserializeSimulation(serializedData);

        _simController->updateData(deserializedData.content);
*/
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenSimulationDialog::show()
{
    ifd::FileDialog::Instance().Open(
        "SimulationOpenDialog", "Open simulation", "Simulation file (*.sim){.sim},.*", false);
}
