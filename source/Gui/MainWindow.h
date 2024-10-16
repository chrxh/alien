#pragma once

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "Network/Definitions.h"

#include "Definitions.h"

class _MainWindow
{
public:
    _MainWindow(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade, GuiLogger const& logger);
    void mainLoop();
    void shutdown();

private:
    char const* initGlfwAndReturnGlslVersion();

    void processLoadingScreen();
    void processFadeoutLoadingScreen();
    void processFadeInControls();
    void processReady();

    void renderSimulation();

    void processMenubar();
    void processDialogs();
    void processWindows();
    void processControllers();

    void onRunSimulation();
    void onPauseSimulation();
    void onExit();

    void finishFrame();

    void pushGlobalStyle();
    void popGlobalStyle();

    GLFWwindow* _window;
    GuiLogger _logger;

    SimulationView _simulationView;
    TemporalControlWindow _temporalControlWindow;
    SpatialControlWindow _spatialControlWindow;
    SimulationParametersWindow _simulationParametersWindow;
    StatisticsWindow _statisticsWindow;
    LogWindow _logWindow;
    GettingStartedWindow _gettingStartedWindow;
    ShaderWindow _shaderWindow;
    RadiationSourcesWindow _radiationSourcesWindow;
    AutosaveWindow _autosaveWindow;

    ExitDialog _exitDialog;
    GpuSettingsDialog _gpuSettingsDialog;
    MassOperationsDialog _massOperationsDialog;
    NewSimulationDialog _newSimulationDialog;
    DisplaySettingsDialog _displaySettingsDialog;
    PatternAnalysisDialog _patternAnalysisDialog;
    AboutDialog _aboutDialog;
    CreateUserDialog _createUserDialog;
    UploadSimulationDialog _uploadSimulationDialog;
    EditSimulationDialog _editSimulationDialog;
    ActivateUserDialog _activateUserDialog;
    DeleteUserDialog _deleteUserDialog;
    NetworkSettingsDialog _networkSettingsDialog;
    ResetPasswordDialog _resetPasswordDialog;
    NewPasswordDialog _newPasswordDialog;
    ImageToPatternDialog _imageToPatternDialog;

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;
    SimulationInteractionController _simInteractionController;
    StartupController _startupController;
    AutosaveController _autosaveController; 
    UiController _uiController; 
    EditorController _editorController; 
    FpsController _fpsController;

    bool _onExit = false;
    bool _simulationMenuToggled = false;
    bool _networkMenuToggled = false;
    bool _windowMenuToggled = false;
    bool _settingsMenuToggled = false;
    bool _viewMenuToggled = false;
    bool _editorMenuToggled = false;
    bool _toolsMenuToggled = false;
    bool _helpMenuToggled = false;
    bool _renderSimulation = true;
};