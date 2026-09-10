#pragma once

#include <chrono>
#include <optional>

#include <Base/Singleton.h>

#include "Definitions.h"

class ConsoleModeController
{
    MAKE_SINGLETON(ConsoleModeController);

public:
    void activate();
    bool isActive() const;

    void process();  // Must not be called within an ImGui frame.

private:
    void deactivate();
    void printPersistedSavepoint();
    void printStatusLine();

    struct StateBeforeActivation
    {
        bool syncSimulationWithRendering = false;
        std::optional<int> tpsRestriction;
    };

    bool _active = false;
    StateBeforeActivation _stateBeforeActivation;
    std::optional<std::chrono::steady_clock::time_point> _lastPrintTimepoint;
};
