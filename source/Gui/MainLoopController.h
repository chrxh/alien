#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"

#include "Definitions.h"

class MainLoopController
{
    MAKE_SINGLETON(MainLoopController);

public:
    void setup(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade);
    void process();
    void scheduleShutdown();

    bool shouldClose() const;

private:
    void processFirstTick();
    void processLoadingScreen();
    void processFadeOutLoadingScreen();
    void processFadeInUI();
    void processOperatingMode();
    void processScheduleExit();
    void processExiting();

    void drawLoadingScreen();
    void decreaseAlphaForFadeOutLoadingScreen();
    void increaseAlphaForFadeInUI();
    void processMenubar();

    void pushGlobalStyle();
    void popGlobalStyle();

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade;

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

    PersisterRequestId _loadSimRequestId;
    PersisterRequestId _saveSimRequestId;

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
