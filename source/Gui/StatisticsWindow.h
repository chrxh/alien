#pragma once

#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _StatisticsWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);

    void reset();
    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void processLiveStatistics();
    void processLongtermStatistics();

    void updateData();

    SimulationController _simController;

    bool _on = true;
    bool _live = true;

    struct LiveStatistics
    {
        float timepoint = 0.0f; //in seconds
        float history = 10.0f;  //in seconds
        std::vector<float> timepointsHistory;
        std::vector<float> numCellsHistory;
        std::vector<float> numParticlesHistory;
        std::vector<float> numTokensHistory;
    };
    LiveStatistics _liveStatistics;

    struct LongtermStatistics
    {
        std::vector<float> timestepHistory;
        std::vector<float> numCellsHistory;
        std::vector<float> numParticlesHistory;
        std::vector<float> numTokensHistory;
    };
    LongtermStatistics _longtermStatistics;
};
