#pragma once

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
};