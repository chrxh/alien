#pragma once

#include <chrono>
#include <deque>
#include <optional>

#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>
#include <Data/Interface/SimulationParameters.h>

class TemporalControlService
{
    MAKE_SINGLETON(TemporalControlService);

public:
    static auto constexpr MaxNumPreviousTimesteps = 20;

    void calcSingleTimestep();
    bool hasPreviousTimestep();
    void restorePreviousTimestep();
    int getNumPreviousTimesteps();
    void clearPreviousTimesteps();

    void createFlashback();
    bool hasFlashback() const;
    void restoreFlashback();

private:
    struct Snapshot
    {
        uint64_t timestep = 0;
        std::chrono::milliseconds realTime;
        SimulationParameters parameters;
        ContentDesc data;
    };
    Snapshot createSnapshot() const;
    void applySnapshot(Snapshot const& snapshot) const;
    void restorePosition(RealVector2D& position, RealVector2D const& velocity, RealVector2D const& origPosition, RealVector2D const& origVelocity) const;

    void clearPreviousTimestepsOfOtherSessions();

    std::optional<Snapshot> _flashback;
    std::deque<Snapshot> _previousTimesteps;
    std::optional<int> _sessionId;
};
