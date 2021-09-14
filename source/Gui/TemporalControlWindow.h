#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _TemporalControlWindow
{
public:
    _TemporalControlWindow(SimulationController const& simController, StyleRepository const& styleRepository);

    void process();

private:
    void processTpsInfo();
    void processTotalTimestepsInfo();
    void processTpsRestriction();

    void processRunButton();
    void processPauseButton();
    void processStepBackwardButton();
    void processStepForwardButton();
    void processSnapshotButton();
    void processRestoreButton();

    SimulationController _simController; 
    StyleRepository _styleRepository;

    TextureData _runTexture;
    TextureData _pauseTexture;
    TextureData _stepBackwardTexture;
    TextureData _stepForwardTexture;
    TextureData _snapshotTexture;
    TextureData _restoreTexture;

    struct Snapshot
    {
        uint64_t timestep;
        DataDescription data;
    };
    boost::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;
};