#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _MainWindow
{
public:
    _MainWindow(SimulationController const& simController, SimpleLogger const& logger);
    void mainLoop();
    void shutdown();

private:
    char const* initGlfw();  //return glsl version

    void processUninitialized();
    void processRequestLoading();
    void processLoadingSimulation();
    void processLoadingControls();
    void processFinishedLoading();

    void renderSimulation();

    void processMenubar();
    void processDialogs();
    void processWindows();
    void processControllers();

    void onRunSimulation();
    void onPauseSimulation();

    void processExitDialog();
    void reset();

    GLFWwindow* _window;
    SimpleLogger _logger;

    Viewport _viewport;

    SimulationView _simulationView;
    TemporalControlWindow _temporalControlWindow;
    SpatialControlWindow _spatialControlWindow;
    SimulationParametersWindow _simulationParametersWindow;
    StatisticsWindow _statisticsWindow;
    FlowGeneratorWindow _flowGeneratorWindow;
    LogWindow _logWindow;
    GettingStartedWindow _gettingStartedWindow;
    BrowserWindow _browserWindow;

    GpuSettingsDialog _gpuSettingsDialog;
    ColorizeDialog _colorizeDialog;
    NewSimulationDialog _newSimulationDialog;
    OpenSimulationDialog _openSimulationDialog; 
    SaveSimulationDialog _saveSimulationDialog; 
    DisplaySettingsDialog _displaySettingsDialog;
    PatternAnalysisDialog _patternAnalysisDialog;
    AboutDialog _aboutDialog;

    ModeController _modeController;
    WindowController _windowController;
    SimulationController _simController;
    StartupController _startupController;
    AutosaveController _autosaveController; 
    UiController _uiController; 
    EditorController _editorController; 
    FpsController _fpsController;

    bool _onClose = false;
    bool _simulationMenuToggled = false;
    bool _networkMenuToggled = false;
    bool _windowMenuToggled = false;
    bool _settingsMenuToggled = false;
    bool _viewMenuToggled = false;
    bool _editorMenuToggled = false;
    bool _toolsMenuToggled = false;
    bool _helpMenuToggled = false;
    bool _showExitDialog = false;
    bool _renderSimulation = true;
};