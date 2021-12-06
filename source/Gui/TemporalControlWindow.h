#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _TemporalControlWindow
{
public:
    _TemporalControlWindow(
        SimulationController const& simController,
        StyleRepository const& styleRepository,
        StatisticsWindow const& statisticsWindow);
    ~_TemporalControlWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

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
    StatisticsWindow _statisticsWindow;

    TextureData _runTexture;
    TextureData _pauseTexture;
    TextureData _stepBackwardTexture;
    TextureData _stepForwardTexture;
    TextureData _snapshotTexture;
    TextureData _restoreTexture;

    struct Snapshot
    {
        uint64_t timestep;
        DataDescription2 data;
    };
    boost::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;
    bool _on = false;

    bool _slowDown = false;
    int _tpsRestriction = 30;
};