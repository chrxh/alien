#pragma once

#include <chrono>

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>
#include <EngineInterface/SimulationParameters.h>

#include "AlienWindow.h"
#include "Definitions.h"

class TemporalControlWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TemporalControlWindow);

public:
    void onSnapshot();

private:
    TemporalControlWindow();

    void initIntern() override;
    void processIntern();

    void processTpsInfo();
    void processTotalTimestepsInfo();
    void processRealTimeInfo();
    void processTpsRestriction();

    void processRunButton();
    void processPauseButton();
    void processStepBackwardButton();
    void processStepForwardButton();
    void processCreateFlashbackButton();
    void processLoadFlashbackButton();

    struct Snapshot
    {
        uint64_t timestep;
        std::chrono::milliseconds realTime;
        SimulationParameters parameters;
        Description data;
    };
    Snapshot createSnapshot();
    void applySnapshot(Snapshot const& snapshot);

    void restorePosition(RealVector2D& position, RealVector2D const& velocity, RealVector2D const& origPosition, RealVector2D const& origVelocity);

    std::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;

    bool _slowDown = false;
    int _tpsRestriction = 100;

    std::optional<int> _sessionId;
};
