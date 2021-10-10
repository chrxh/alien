#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

struct GLFWvidmode;
class _MainWindow
{
public:
    GLFWwindow* init(SimulationController const& simController);
    void mainLoop(GLFWwindow* window);
    void shutdown(GLFWwindow* window);

private:
    struct GlfwData
    {
        GLFWwindow* window;
        GLFWvidmode const* mode;
        char const* glsl_version;
    };
    GlfwData initGlfw();

    void processMenubar();
    void processDialogs();
    void processWindows();

    void onOpenSimulation();
    void onSaveSimulation();
    void onRunSimulation();
    void onPauseSimulation();

    void reset();

    SimulationController _simController;
    SimulationView _simulationView;
    TemporalControlWindow _temporalControlWindow;
    SpatialControlWindow _spatialControlWindow;
    SimulationParametersWindow _simulationParametersWindow;
    StatisticsWindow _statisticsWindow;
    ModeWindow _modeWindow;
    GpuSettingsWindow _gpuSettingsWindow;
    Viewport _viewport;
    NewSimulationDialog _newSimulationDialog;
    StartupWindow _startupWindow;

    StyleRepository _styleRepository;

    bool _onClose = false;
};