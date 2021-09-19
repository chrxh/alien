#pragma once

#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _MonitorWindow
{
public:
    _MonitorWindow(SimulationController const& simController);

    void process();

private:
    void updateData();

    SimulationController _simController;

    bool _on = true;
    bool _live = true;

    float _timepoint = 0.0f;
    float _history = 10.0f;
    std::vector<float> _timepointsHistory;
    std::vector<float> _numCellsHistory;
    std::vector<float> _numParticlesHistory;
    std::vector<float> _numTokensHistory;
};
