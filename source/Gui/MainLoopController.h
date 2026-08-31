#pragma once

#include <chrono>

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Descs.h>

#include <Network/NetworkResourceRawTO.h>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/PersisterRequestId.h>

#include "Definitions.h"

class MainLoopController
{
    MAKE_SINGLETON(MainLoopController);

public:
    void setup();
    void process();
    void shutdown();

    void scheduleClosing();
    bool shouldClose() const;

private:
    void processFirstTick();
    void processLoadingScreen();
    void processFadeOutLoadingScreen();
    void processFadeInUI();
    void processOperatingMode();
    void processScheduleExit();
    void processExiting();

    void scheduleReadingAutosave();
    void scheduleSearchingStartupSimulation();
    void scheduleDownloadingStartupSimulation(NetworkResourceRawTO const& resourceTO);
    void setupSimulation(SimulationDesc const& simulationDesc);
    void setupEmptySimulation();
    void finishSimulationLoading(SimulationDesc const& simulationDesc);

    void drawLoadingScreen();
    void decreaseAlphaForFadeOutLoadingScreen();
    void increaseAlphaForFadeInUI();
    void processMenubar();

    void pushGlobalStyle();
    void popGlobalStyle();

    enum class ProgramState
    {
        FirstTick,
        LoadingScreen,
        FadeOutLoadingScreen,
        FadeInUI,
        OperatingMode,
        ScheduleExit,
        Exiting,
        Finished
    };
    ProgramState _programState = ProgramState::FirstTick;
    bool _saveOnExit = true;

    TaskProcessor _startupProcessor;
    TaskProcessor _downloadProcessor;
    PersisterRequestId _saveSimRequestId;
    std::string _loadedSimulationName;

    TextureData _logo;
    std::optional<std::chrono::steady_clock::time_point> _simulationLoadedTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _fadedOutTimepoint;

    bool _simulationMenuOpened = false;
    bool _networkMenuOpened = false;
    bool _windowMenuOpened = false;
    bool _settingsMenuOpened = false;
    bool _viewMenuOpened = false;
    bool _editorMenuOpened = false;
    bool _toolsMenuOpened = false;
    bool _helpMenuOpened = false;
};
