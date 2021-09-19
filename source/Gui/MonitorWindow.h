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

    int _offset = 0;
    std::vector<float> _timestepsHistory;
    std::vector<float> _numCellsHistory;
    std::vector<float> _numParticlesHistory;
    std::vector<float> _numTokensHistory;
    bool _on = true;
};
